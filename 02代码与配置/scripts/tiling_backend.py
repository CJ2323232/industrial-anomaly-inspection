"""Native-resolution overlapping tiles; no resizing or GT-guided crop selection."""
from collections import namedtuple

import numpy as np
import torch

Prediction = namedtuple('Prediction', 'pred_score anomaly_map')


def starts(length, tile=256, stride=224):
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (length, tile, stride)):
        raise ValueError('Integer geometry required')
    if not 1 <= stride <= tile <= length:
        raise ValueError('Require 1 <= stride <= tile <= image length')
    result = list(range(0, length - tile + 1, stride))
    if result[-1] != length - tile:
        result.append(length - tile)
    return result


def windows(height, width, tile=256, stride=224):
    return [(y, x, y + tile, x + tile)
            for y in starts(height, tile, stride) for x in starts(width, tile, stride)]


def blend_weight(tile, device, dtype=torch.float32):
    # Positive endpoints preserve image borders; lower weights reduce seam influence.
    axis = torch.linspace(-1, 1, tile, device=device, dtype=dtype)
    axis = (1 - axis.abs()).clamp_min(.05)
    return axis[:, None] * axis[None, :]


class OverlapTiler:
    """Sequential batch1 tile forwards, max weighted tile score, blended maps.

    Image score is max(reference PatchCore weighted tile score), NOT the maximum
    blended-map pixel. It is independently calibrated on whole normal images.
    """
    def __init__(self, model, height=700, width=700, tile=256, stride=224):
        self.model = model
        self.height, self.width, self.tile = height, width, tile
        self.windows = windows(height, width, tile, stride)
        device = model.memory_bank.device
        self.weight = blend_weight(tile, device)
        self.denominator = torch.zeros((height, width), device=device)
        for y, x, yy, xx in self.windows:
            self.denominator[y:yy, x:xx] += self.weight
        if not bool(torch.all(self.denominator > 0)):
            raise ValueError('Uncovered image coordinates')

    def __call__(self, image):
        if tuple(image.shape) != (1, 3, self.height, self.width):
            raise ValueError('Expected one original-resolution RGB image')
        accumulator = torch.zeros_like(self.denominator)
        scores = []
        for y, x, yy, xx in self.windows:
            output = self.model(image[:, :, y:yy, x:xx].contiguous())
            if tuple(output.anomaly_map.shape) != (1, 1, self.tile, self.tile):
                raise ValueError('Reference model returned an unexpected tile map')
            accumulator[y:yy, x:xx] += output.anomaly_map[0, 0] * self.weight
            scores.append(output.pred_score.reshape(()))
        return Prediction(torch.stack(scores).amax().reshape(1),
                          (accumulator / self.denominator)[None, None])


def sampled_patch_indices(tile_count, patches_per_tile=1024, budget=1024, seed=0):
    """Balance a fixed per-image candidate budget across all tiles, without replacement."""
    if not 1 <= tile_count <= budget <= tile_count * patches_per_tile:
        raise ValueError('Invalid per-image candidate budget')
    rng = np.random.default_rng(seed)
    counts = np.full(tile_count, budget // tile_count, dtype=int)
    counts[rng.choice(tile_count, budget % tile_count, replace=False)] += 1
    return [sorted(rng.choice(patches_per_tile, int(n), replace=False).tolist()) for n in counts]
