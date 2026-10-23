"""Temporarily change only reference PatchCore anomaly-map smoothing."""
from contextlib import contextmanager


@contextmanager
def use_smoothing(model, sigma):
    """Install a reference generator; preserve the original object and device.

    Setup/teardown is outside benchmark timing. The inactive original generator
    stays on CPU, so GPU allocation includes only the active generator.
    """
    if isinstance(sigma, bool) or not isinstance(sigma, int) or sigma < 1:
        raise ValueError('sigma must be a positive integer')
    from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator
    original = model.anomaly_map_generator
    device = model.memory_bank.device
    candidate = AnomalyMapGenerator(sigma=sigma).to(device).eval()
    model.anomaly_map_generator = candidate
    original.to('cpu')
    try:
        yield model
    finally:
        candidate.to('cpu')
        model.anomaly_map_generator = original.to(device)
