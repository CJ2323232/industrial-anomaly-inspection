"""Exact 1-nearest-neighbor search with bounded distance-matrix blocks.

This learning implementation uses PyTorch cdist for Euclidean distances and
implements the block traversal and running minimum independently of anomalib.
No approximate index, coreset selection or image-level weighting is performed.
"""

import torch


@torch.no_grad()
def exact_nearest_neighbors(queries, bank, query_chunk=128, bank_chunk=256):
    """Return minimum distances and bank row indices for N query vectors.

Both inputs must be finite, nonempty 2D float32/float64 tensors on the same
device, with the same dtype and feature dimension. All bank rows are searched.
Equal computed distances select the smallest bank index. cdist uses direct
differences, avoiding norm-expansion cancellation near identical vectors.
This limits distance-block storage; it does not assert lower total latency.
"""
    for value in (query_chunk, bank_chunk):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Chunk sizes must be positive integers")
    if queries.ndim != 2 or bank.ndim != 2 or min(*queries.shape, *bank.shape) < 1:
        raise ValueError("Queries and bank must be nonempty 2D matrices")
    if queries.shape[1] != bank.shape[1]:
        raise ValueError("Feature dimensions differ")
    if queries.dtype != bank.dtype or queries.dtype not in (torch.float32, torch.float64):
        raise ValueError("Use matching float32 or float64 dtypes")
    if queries.device != bank.device:
        raise ValueError("Queries and bank must be on the same device")
    if not torch.isfinite(queries).all() or not torch.isfinite(bank).all():
        raise ValueError("Features must be finite")
    distances = torch.empty(queries.shape[0], dtype=queries.dtype, device=queries.device)
    indices = torch.empty(queries.shape[0], dtype=torch.long, device=queries.device)
    for start in range(0, queries.shape[0], query_chunk):
        query = queries[start:start + query_chunk]
        best = torch.full((query.shape[0],), float("inf"), dtype=query.dtype, device=query.device)
        best_index = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        for offset in range(0, bank.shape[0], bank_chunk):
            block = torch.cdist(query, bank[offset:offset + bank_chunk], p=2,
                                compute_mode="donot_use_mm_for_euclid_dist")
            candidate, local_index = block.min(dim=1)
            improved = candidate < best
            best = torch.where(improved, candidate, best)
            best_index = torch.where(improved, local_index + offset, best_index)
            # Release the block before allocating the next one.
            del block
        distances[start:start + query.shape[0]] = best
        indices[start:start + query.shape[0]] = best_index
    if not torch.isfinite(distances).all():
        raise ValueError("Distances overflowed; inspect input scale or use float64")
    return distances, indices
