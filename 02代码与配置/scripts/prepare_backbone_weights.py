"""Download one official torchvision weight file into the project, or verify it offline."""
import argparse
import json
from pathlib import Path

if __package__:
    from .backbone_backend import WEIGHT_NAME, WEIGHT_PREFIX, WEIGHT_URL, verify_weight_file
else:
    from backbone_backend import WEIGHT_NAME, WEIGHT_PREFIX, WEIGHT_URL, verify_weight_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    path = root / '.cache/models/torch/hub/checkpoints' / WEIGHT_NAME
    metadata = root / '02代码与配置/configs/pretrained/wide-resnet50-tv-in1k.json'
    previous = json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else None
    if not path.exists():
        if not args.download:
            raise FileNotFoundError('Run prepare_backbone_weights.py --download once to cache the official weights')
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHT_URL, str(path), hash_prefix=WEIGHT_PREFIX, progress=True)
    record = verify_weight_file(path, previous['sha256'] if previous else None)
    if not previous:
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
