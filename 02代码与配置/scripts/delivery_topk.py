"""Frozen Top-11 aggregation for delivery; matches archived group C arithmetic."""
import numpy as np
def topk_image_score(patch_distances,k=11,expected_patches=1024):
    if k!=11 or expected_patches!=1024:
        raise ValueError('当前C候选只接受已冻结的1024个Patch与K=11。')
    values=np.asarray(patch_distances)
    if values.size!=expected_patches or not np.issubdtype(values.dtype,np.floating) or not np.isfinite(values).all():
        raise ValueError('C评分需要1024个有限的浮点Patch距离。')
    flat=values.astype(np.float64).reshape(-1)
    indices=np.lexsort((np.arange(expected_patches),-flat))[:k]
    return float(flat[indices].mean()),dict(K=k,patch_count=expected_patches,top_indices=indices.tolist(),
        accumulation='float64 mean of raw float32 patch distances',max_patch_distance=float(flat.max()),
        rule='fixed ceil(1024 * 0.01) = 11; no neighbor weight')
