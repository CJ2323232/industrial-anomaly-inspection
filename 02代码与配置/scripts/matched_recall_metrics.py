"""Exact empirical recall cutoffs for saved float32 anomaly maps.

One global cutoff per model; never optimize each image separately. Ties are
included together, so actual recall may exceed the requested minimum.
"""
import math
from numbers import Integral

import numpy as np


def sorted_scores(values):
    values = np.asarray(values)
    if values.dtype != np.float32 or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Scores must be a finite one-dimensional float32 array')
    return np.sort(values)


def cutoff_for_tp(positive_sorted, required_tp):
    """Highest float32 cutoff with at least required_tp positives under score > t.

Input must be ascending, finite float32 positive-label scores. Full validation
belongs to sorted_scores; avoid rechecking millions of values per curve point.
"""
    if (not isinstance(required_tp, Integral) or isinstance(required_tp, bool)
            or not 1 <= required_tp <= len(positive_sorted)):
        raise ValueError('required_tp must be an integer in [1, number of positives]')
    if positive_sorted.dtype != np.float32 or positive_sorted.ndim != 1:
        raise ValueError('Expected sorted float32 scores')
    boundary = positive_sorted[len(positive_sorted) - required_tp]
    threshold = np.nextafter(boundary, np.float32(-np.inf))
    if not np.isfinite(threshold):
        raise ValueError('No finite float32 cutoff can include this boundary')
    return float(threshold)


def counts_at(positive_sorted, negative_sorted, threshold):
    """Exact pixel counts, matching NumPy float32 map comparisons and strict >."""
    if not math.isfinite(threshold):
        raise ValueError('Threshold must be finite')
    threshold = np.float32(threshold)
    fn = int(np.searchsorted(positive_sorted, threshold, side='right'))
    tn = int(np.searchsorted(negative_sorted, threshold, side='right'))
    return {'TP': len(positive_sorted) - fn, 'FP': len(negative_sorted) - tn,
            'FN': fn, 'TN': tn}
