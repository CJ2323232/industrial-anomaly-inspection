"""Switch the existing PatchCore inference pipeline between retrieval backends.

The blocked backend uses the same norm-expansion Euclidean formula and dtype
as anomalib 2.2.0. It supports both patch 1-NN and weighted-score support k-NN.
Feature extraction, bank, weighting and map generation remain reference code.
"""
from contextlib import contextmanager
from types import MethodType

import torch


@torch.no_grad()
def blocked_knn(queries, bank, k=1, query_chunk=128, bank_chunk=512):
    """Exhaustive k-NN over finite 2D floating tensors, bounded matrix blocks.

Caller owns finite-value validation to avoid device synchronization inside each
timed search. k=1 returns vectors; k>1 returns sorted N-by-k matrices. Exact
computed ties use the first index for k=1; top-k tie order is not prescribed.
"""
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (k, query_chunk, bank_chunk)):
        raise ValueError("k and chunks must be positive integers")
    if queries.ndim != 2 or bank.ndim != 2 or min(*queries.shape, *bank.shape) < 1:
        raise ValueError("Expected nonempty 2D features")
    if queries.shape[1] != bank.shape[1] or k > bank.shape[0]:
        raise ValueError("Invalid feature dimensions or k larger than bank")
    if queries.dtype != bank.dtype or queries.dtype not in (torch.float32, torch.float64) or queries.device != bank.device:
        raise ValueError("Expected matching float32/float64 dtype and device")
    output_shape = (queries.shape[0],) if k == 1 else (queries.shape[0], k)
    output_d = torch.empty(output_shape, device=queries.device, dtype=queries.dtype)
    output_i = torch.empty(output_shape, device=queries.device, dtype=torch.long)
    for start in range(0, queries.shape[0], query_chunk):
        query = queries[start:start + query_chunk]
        query_norm = query.pow(2).sum(-1, keepdim=True)
        best_d = best_i = None
        for offset in range(0, bank.shape[0], bank_chunk):
            subset = bank[offset:offset + bank_chunk]
            matrix = query_norm - 2 * (query @ subset.T) + subset.pow(2).sum(-1).unsqueeze(0)
            matrix.clamp_min_(0).sqrt_()
            if k == 1:
                candidate_d, candidate_i = matrix.min(dim=1)
                candidate_i = candidate_i + offset
                if best_d is None:
                    best_d, best_i = candidate_d, candidate_i
                else:
                    improve = candidate_d < best_d
                    best_d = torch.where(improve, candidate_d, best_d)
                    best_i = torch.where(improve, candidate_i, best_i)
            else:
                candidate_d, candidate_i = matrix.topk(min(k, subset.shape[0]), dim=1, largest=False, sorted=True)
                candidate_i = candidate_i + offset
                if best_d is None:
                    best_d, best_i = candidate_d, candidate_i
                else:
                    combined_d = torch.cat((best_d, candidate_d), dim=1)
                    combined_i = torch.cat((best_i, candidate_i), dim=1)
                    best_d, positions = combined_d.topk(min(k, combined_d.shape[1]), dim=1, largest=False, sorted=True)
                    best_i = combined_i.gather(1, positions)
                    del combined_d, combined_i, positions
            del matrix
        output_d[start:start + query.shape[0]] = best_d
        output_i[start:start + query.shape[0]] = best_i
    return output_d, output_i


@contextmanager
def use_retrieval(model, backend, query_chunk=128, bank_chunk=512, trace=None):
    """Temporarily change retrieval; restore the original method even on error.

Optional CPU traces are only for untimed correctness runs, never benchmarks.
"""
    if backend not in ("reference", "blocked"):
        raise ValueError("Unknown retrieval backend")
    had_instance_method = "nearest_neighbors" in model.__dict__
    original = model.nearest_neighbors

    def search(instance, embedding, n_neighbors):
        if backend == "reference":
            distances, indices = original(embedding, n_neighbors)
        else:
            distances, indices = blocked_knn(embedding, instance.memory_bank, n_neighbors, query_chunk, bank_chunk)
        if trace is not None:
            trace.append({"k": n_neighbors, "distances": distances.detach().cpu().numpy().copy(),
                          "indices": indices.detach().cpu().numpy().copy()})
        return distances, indices

    if backend == "reference" and trace is None:
        yield model
        return
    model.nearest_neighbors = MethodType(search, model)
    try:
        yield model
    finally:
        if had_instance_method:
            model.nearest_neighbors = original
        else:
            delattr(model, "nearest_neighbors")
