"""Learning implementation of local pooling, fusion and spatial flattening.

Algorithm reference: anomalib 2.2.0 PatchcoreModel. This module implements only
feature organization in PyTorch; it is not a complete independent PatchCore.
"""

import torch
from torch.nn import functional as F


def organize_features(layer2, layer3):
    """Return pooled layers, aligned layer3, fused map and spatial patch rows."""
    if layer2.ndim != 4 or layer3.ndim != 4 or layer2.shape[0] != layer3.shape[0]:
        raise ValueError("Features must be NCHW tensors with matching batch sizes")
    pooled2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
    pooled3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
    aligned3 = F.interpolate(pooled3, size=pooled2.shape[-2:], mode="bilinear", align_corners=False)
    fused = torch.cat((pooled2, aligned3), dim=1)
    # Each row represents one spatial position; all channels stay in that row.
    patches = fused.permute(0, 2, 3, 1).reshape(-1, fused.shape[1])
    return {"pooled_layer2": pooled2, "pooled_layer3": pooled3,
            "aligned_layer3": aligned3, "fused_map": fused, "patch_vectors": patches}
