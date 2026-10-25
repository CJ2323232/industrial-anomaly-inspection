"""Strict torchvision ImageNet weight loading into the reference timm feature extractor."""
from pathlib import Path

if __package__:
    from .baseline_utils import file_sha256
else:
    from baseline_utils import file_sha256

MODEL_NAME = 'wide_resnet50_2.tv_in1k'
WEIGHT_NAME = 'wide_resnet50_2-95faca4d.pth'
WEIGHT_URL = 'https://download.pytorch.org/models/' + WEIGHT_NAME
WEIGHT_PREFIX = '95faca4d'


def verify_weight_file(path, expected_sha256=None):
    path = Path(path)
    digest = file_sha256(path)
    if not digest.startswith(WEIGHT_PREFIX) or (expected_sha256 is not None and digest != expected_sha256):
        raise ValueError('WideResNet50 pretrained weight identity mismatch')
    return {'filename': WEIGHT_NAME, 'url': WEIGHT_URL, 'sha256': digest, 'bytes': path.stat().st_size,
            'source': 'torchvision ImageNet-1K V1; timm wide_resnet50_2.tv_in1k mapping'}


def select_feature_weights(target, source):
    """No missing feature weights, mismatched shapes or unexpected ignored keys."""
    missing = set(target) - set(source)
    unused = set(source) - set(target)
    unexpected = {key for key in unused if not key.startswith(('layer4.', 'fc.'))}
    if missing or unexpected:
        raise ValueError(f'Pretrained mapping incomplete: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    if any(target[key].shape != source[key].shape for key in target):
        raise ValueError('Pretrained feature tensor shape mismatch')
    return {key: source[key] for key in target}, sorted(unused)


def load_pretrained_features(model, weight_path):
    import torch
    source = torch.load(weight_path, map_location='cpu', weights_only=True)
    module = model.feature_extractor.feature_extractor
    selected, unused = select_feature_weights(module.state_dict(), source)
    module.load_state_dict(selected, strict=True)
    if not all(torch.equal(value, source[key]) for key, value in module.state_dict().items()):
        raise ValueError('Loaded backbone does not reproduce pretrained tensors')
    return {'loaded_feature_tensors': len(selected), 'ignored_tail_tensors': unused,
            'strict_feature_load': True, 'loaded_tensors_equal_source': True}
