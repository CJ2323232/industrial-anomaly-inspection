"""Pixel-pooled threshold diagnostics; no model or dataset I/O."""
import numpy as np


def pixel_metrics(counts):
    tp, fp, fn, tn = (counts[k] for k in ('TP', 'FP', 'FN', 'TN'))
    if any(not isinstance(v, (int, np.integer)) or v < 0 for v in (tp, fp, fn, tn)):
        raise ValueError('Pixel counts must be nonnegative integers')
    ratio = lambda n, d: n / d if d else None
    return {**counts, 'Pixel_Recall': ratio(tp, tp + fn),
            'Pixel_IoU': ratio(tp, tp + fp + fn), 'Pixel_Precision': ratio(tp, tp + fp),
            'FP_area_fraction': ratio(fp, tp + fp + fn + tn),
            'normal_pixel_FPR': ratio(fp, fp + tn),
            'predicted_contamination': ratio(fp, tp + fp)}


def rank_metrics(positive_sorted, negative_sorted):
    """Exact AP and AUROC including ties, from ascending labeled score arrays.

The AP step changes only where positive scores occur; negative-only score
levels contribute zero recall increment. Normal-only images remain undefined.
"""
    p, n = len(positive_sorted), len(negative_sorted)
    if not p:
        return {'AP': None, 'AUROC': None}
    levels, counts = np.unique(positive_sorted, return_counts=True)
    tp = np.cumsum(counts[::-1], dtype=np.int64)[::-1]
    lower = np.searchsorted(negative_sorted, levels, side='left')
    upper = np.searchsorted(negative_sorted, levels, side='right')
    fp = n - lower
    ap = float(np.sum((counts / p) * (tp / (tp + fp))))
    auc = float(np.sum(counts * (lower + (upper - lower) * .5)) / (p * n)) if n else None
    return {'AP': ap, 'AUROC': auc}


def cutoff_at_count(positive_sorted, negative_sorted, required_tp):
    """Minimum-recall working point, with an explicit empty-prediction endpoint."""
    if __package__:
        from .matched_recall_metrics import cutoff_for_tp
    else:
        from matched_recall_metrics import cutoff_for_tp
    if required_tp == 0:
        return float(max(positive_sorted[-1] if len(positive_sorted) else -np.inf,
                         negative_sorted[-1] if len(negative_sorted) else -np.inf))
    return cutoff_for_tp(positive_sorted, required_tp)


def oracle_image_iou(positive_sorted, negative_sorted):
    """Exact per-image threshold upper bound; diagnostic, unavailable at deployment.

Negative-only score levels cannot increase IoU, so checking every positive
score level is exhaustive for a single threshold, with tied scores kept whole.
"""
    if not len(positive_sorted):
        return {'max_IoU': None, 'threshold': None}
    levels, counts = np.unique(positive_sorted, return_counts=True)
    tp = np.cumsum(counts[::-1], dtype=np.int64)[::-1]
    fp = len(negative_sorted) - np.searchsorted(negative_sorted, levels, side='left')
    iou = tp / (len(positive_sorted) + fp)
    best = np.flatnonzero(iou == np.max(iou))[-1]
    threshold = float(np.nextafter(levels[best], np.float32(-np.inf)))
    return {'max_IoU': float(iou[best]), 'threshold': threshold}
