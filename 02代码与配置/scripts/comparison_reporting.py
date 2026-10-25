"""Offline original-coordinate comparison figures for named model configurations."""
import html
from pathlib import Path

import numpy as np

if __package__:
    from .baseline_utils import safe_file
    from .run_trusted_baseline import load_mask, write_json
else:
    from baseline_utils import safe_file
    from run_trusted_baseline import load_mask, write_json


def visualize(output, names, roots, rows, thresholds, manifest, data_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    lookup = {name: {r['image_path']: r for r in rows[name]} for name in names}
    index = []
    for path in manifest['splits']['development']:
        rec = manifest['files'][path]
        mask = load_mask(data_root, rec).astype(bool)
        with Image.open(safe_file(data_root, path)) as im:
            original = np.asarray(im)
        fig, axes = plt.subplots(len(names), 4, figsize=(14, 3.5 * len(names)), squeeze=False)
        for n, name in enumerate(names):
            row = lookup[name][path]
            with np.load(safe_file(roots[name], row['map_path'])) as archive:
                array = archive['anomaly_map']
            tau = thresholds[name]['pixel_threshold']
            predicted = array > tau
            axes[n, 0].imshow(original)
            axes[n, 1].imshow(array / tau, cmap='turbo', vmin=0, vmax=2)
            axes[n, 2].imshow(predicted, cmap='gray', vmin=0, vmax=1)
            for col in (0, 2):
                if mask.any():
                    axes[n, col].contour(mask, levels=[.5], colors=['yellow'], linewidths=.7)
            counts = row['pixel_counts']
            iou = counts['TP'] / max(counts['TP'] + counts['FP'] + counts['FN'], 1)
            error = np.zeros((*mask.shape, 3), dtype=np.uint8)
            error[mask & predicted] = (0, 180, 0)
            error[~mask & predicted] = (255, 0, 0)
            error[mask & ~predicted] = (0, 100, 255)
            axes[n, 3].imshow(error)
            axes[n, 0].set_title(f'{name}: original / GT yellow', fontsize=10)
            axes[n, 1].set_title('map / own threshold (display only)', fontsize=10)
            axes[n, 2].set_title(f'IoU={iou:.3f}; image pred={row["pred_label"]}', fontsize=10)
            axes[n, 3].set_title(f'Green TP / red FP / blue FN\nFP={counts["FP"]}, FN={counts["FN"]}', fontsize=10)
        for axis in axes.flat:
            axis.axis('off')
        fig.suptitle(path)
        fig.tight_layout()
        destination = output / '可视化' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=100)
        entry = {'image_path': path, 'path': destination.relative_to(output).as_posix()}
        if mask.any() and float(mask.mean()) < .01:
            yy, xx = np.where(mask)
            for axis in axes.flat:
                axis.set_xlim(max(0, xx.min()-20), min(rec['width'], xx.max()+21))
                axis.set_ylim(min(rec['height'], yy.max()+21), max(0, yy.min()-20))
            zoom = destination.with_name(destination.stem + '_zoom.png')
            fig.savefig(zoom, dpi=100)
            entry['zoom_path'] = zoom.relative_to(output).as_posix()
        plt.close(fig)
        index.append(entry)
    write_json(output / 'visualization_index.json', index)
    return index


def write_html(output, title, lines, images):
    cards = []
    for row in images:
        content = f'<img loading="lazy" src="{html.escape(row["path"])}">'
        if 'zoom_path' in row:
            content += f'<p>小面积缺陷放大（全图指标不变）</p><img loading="lazy" src="{html.escape(row["zoom_path"])}">'
        cards.append(f'<details><summary>{html.escape(row["image_path"])}</summary>{content}</details>')
    text = '<!doctype html><meta charset="utf-8"><title>' + html.escape(title) + '</title>'
    text += '<style>body{max-width:1400px;margin:30px auto;padding:0 20px;font:16px/1.6 sans-serif}pre{white-space:pre-wrap}img{width:100%}summary{padding:12px;cursor:pointer}</style>'
    text += '<h1>' + html.escape(title) + '</h1><pre>' + html.escape('\n'.join(lines)) + '</pre>'
    text += '<p>每张图上行为ResNet18，下行为WideResNet50；黄色为人工标注边界。全部45张图均展示，不只挑选成功案例。</p>'
    (output / '对照报告.html').write_text(text + ''.join(cards), encoding='utf-8')
