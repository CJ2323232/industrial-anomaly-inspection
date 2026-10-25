"""Saved-map, development-label-informed backbone diagnostic; no inference.

All thresholds are global per backbone, selected using pooled positive pixels
in the same 45-image development set. These are empirical descriptive points,
not normal-only calibration or unbiased deployment/test measurements.
"""
import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from .baseline_utils import file_sha256, load_manifest, safe_file
    from .matched_recall_metrics import sorted_scores, cutoff_for_tp, counts_at
    from .resolution_metrics import evaluate_records
else:
    from baseline_utils import file_sha256, load_manifest, safe_file
    from matched_recall_metrics import sorted_scores, cutoff_for_tp, counts_at
    from resolution_metrics import evaluate_records

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ID = 'backbone_comparison_20261025_220104_486177'
NAMES = ('resnet18', 'wide_resnet50')
SOURCE = ROOT / '04实验结果/09骨干网络对照' / SOURCE_ID
DATA = ROOT / '03原始数据/解压数据/MVTecAD'


def now():
    return datetime.now(timezone(timedelta(hours=8)))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_inputs(source, data):
    manifest = load_manifest(source / 'split_manifest.json')
    report = read(source / 'report.json')
    if not report['status'].startswith('completed') or report['held_out_images_opened'] != 0:
        raise ValueError('Require completed source experiment with sealed held-out data')
    if manifest['manifest_sha256'] != report['config']['manifest_sha256']:
        raise ValueError('Source split identity mismatch')
    paths = manifest['splits']['development']
    if len(paths) != 45:
        raise ValueError('Expected the fixed 45-image development set')
    identities = {p: file_sha256(source / p) for p in ('report.json', 'config.json', 'split_manifest.json')}
    masks, hashes, originals = {}, {}, {}
    for path in paths:
        record = manifest['files'][path]
        image_path = safe_file(data, path)
        if file_sha256(image_path) != record['image_sha256']:
            raise ValueError('Image identity mismatch: ' + path)
        if record['mask_path']:
            mask_path = safe_file(data, record['mask_path'])
            if file_sha256(mask_path) != record['mask_sha256']:
                raise ValueError('Mask identity mismatch: ' + path)
            with Image.open(mask_path) as image:
                raw = np.asarray(image)
            if not np.isin(raw, [0, 255]).all():
                raise ValueError('Mask must be binary')
            mask = raw > 0
        else:
            mask = np.zeros((record['height'], record['width']), dtype=bool)
        if mask.shape != (record['height'], record['width']) or bool(mask.any()) != bool(record['label']):
            raise ValueError('Mask shape or label mismatch')
        masks[path] = mask
    models = {}
    for name in NAMES:
        directory = source / name
        rows = read(directory / 'development_predictions.json')
        if len(rows) != len(paths) or {r['image_path'] for r in rows} != set(paths):
            raise ValueError('Prediction rows do not exactly match development paths')
        indexed = {r['image_path']: r for r in rows}
        identities[f'{name}/development_predictions.json'] = file_sha256(directory / 'development_predictions.json')
        identities[f'{name}/calibration.json'] = file_sha256(directory / 'calibration.json')
        values = {}
        for index, path in enumerate(paths):
            row, record = indexed[path], manifest['files'][path]
            for key in ('image_sha256', 'mask_sha256', 'label', 'defect_type'):
                if row[key] != record[key]:
                    raise ValueError('Prediction identity mismatch: ' + path)
            expected = 'maps/development/' + str(Path(path).with_suffix('.npz')).replace('\\', '/')
            if row['map_path'] != expected:
                raise ValueError('Only development maps may be consumed')
            map_file = safe_file(directory, row['map_path'])
            hashes[f'{name}/{row["map_path"]}'] = file_sha256(map_file)
            with np.load(map_file, allow_pickle=False) as archive:
                array = archive['anomaly_map']
            if array.shape != masks[path].shape or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError('Expected finite original-resolution float32 map')
            mask = masks[path]
            values[path] = (sorted_scores(array[mask]), sorted_scores(array[~mask]))
            if counts_at(*values[path], report['thresholds'][name]['pixel_threshold']) != row['pixel_counts']:
                raise ValueError('Frozen pixel counts do not reproduce: ' + path)
            if (index + 1) % 15 == 0:
                print(f'validated and sorted {name}: {index + 1}/{len(paths)}', flush=True)
        originals[name] = evaluate_records(rows)
        expected_quality = report['qualities'][name]
        if originals[name] != {key: expected_quality[key] for key in originals[name]}:
            raise ValueError('Frozen quality summary does not reproduce')
        models[name] = {'rows': indexed, 'scores': values,
                        'positive_sorted': np.sort(np.concatenate([v[0] for v in values.values()]))}
    return manifest, report, models, masks, identities, hashes, originals


def evaluate(model, paths, threshold):
    rows = []
    for path in paths:
        row = dict(model['rows'][path])
        row['pixel_counts'] = counts_at(*model['scores'][path], threshold)
        rows.append(row)
    return evaluate_records(rows), rows


def point(models, paths, required_tp, total_pos, key):
    result, rows = {'id': key, 'required_TP': required_tp, 'requested_recall': required_tp / total_pos, 'models': {}}, {}
    for name in NAMES:
        threshold = cutoff_for_tp(models[name]['positive_sorted'], required_tp)
        quality, rows[name] = evaluate(models[name], paths, threshold)
        count = quality['pixel_counts']
        achieved = count['TP'] / total_pos
        if count['TP'] < required_tp:
            raise ValueError('Recall target not achieved')
        macro_recall = float(np.mean([r['pixel_counts']['TP'] / (r['pixel_counts']['TP'] + r['pixel_counts']['FN'])
                                      for r in rows[name] if r['label']]))
        result['models'][name] = {'threshold': threshold, 'actual_recall': achieved,
                                  'macro_recall': macro_recall,
                                  'extra_TP_due_to_ties': count['TP'] - required_tp, 'quality': quality}
    result['actual_recall_gap'] = abs(result['models'][NAMES[0]]['actual_recall'] - result['models'][NAMES[1]]['actual_recall'])
    return result, rows


def flat(point, name):
    row = point['models'][name]
    quality = row['quality']
    return {'threshold': row['threshold'], 'actual_recall': row['actual_recall'],
            'extra_TP_due_to_ties': row['extra_TP_due_to_ties'],
            'FP': quality['pixel_counts']['FP'], 'FN': quality['pixel_counts']['FN'],
            'pooled_IoU': quality['pixel_metrics']['IoU'],
            'macro_IoU': quality['anomaly_macro']['pixel_IoU'],
            'macro_recall': row['macro_recall'],
            'small_IoU': quality['area_groups']['0.1_to_lt_1_percent']['macro']['pixel_IoU'],
            'small_recall': quality['area_groups']['0.1_to_lt_1_percent']['pixel_metrics']['recall'],
            'normal_FP': quality['normal_false_positive_pixels']['FP'],
            'normal_FPR': quality['normal_false_positive_pixels']['rate']}


def draw_curve(output, curve, anchor):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [('FP', 'Extra normal pixels marked defective (all 45 images)'),
             ('macro_IoU', 'Mean IoU across 37 anomalous images'),
             ('normal_FPR', 'False-positive pixel rate on 8 normal images'),
             ('small_recall', 'Pooled recall on 3 small-area anomaly images')]
    for name, color in zip(NAMES, ['#1675b8', '#d66b17']):
        points = [flat(p, name) for p in curve]
        current = flat(anchor, name)
        for axis, (key, label) in zip(axes.flat, specs):
            axis.plot([p['actual_recall'] * 100 for p in points], [p[key] for p in points], label=name, color=color)
            axis.scatter([current['actual_recall'] * 100], [current[key]], color=color, marker='*', s=120, zorder=4)
            axis.set(xlabel='Pooled defect-pixel recall (%)', ylabel=label)
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
            axis.set_xlim(90, 99)
    fig.suptitle('Development-label-informed global cutoffs; stars = frozen ResNet18 recall target')
    fig.tight_layout()
    fig.savefig(output / '召回与误圈权衡.png', dpi=150)
    plt.close(fig)


def visualize(output, source, data, paths, models, masks, anchor, matched_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    index = []
    lookup = {name: {r['image_path']: r for r in matched_rows[name]} for name in NAMES}
    for number, path in enumerate(paths):
        with Image.open(safe_file(data, path)) as image:
            original = np.asarray(image)
        mask = masks[path]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for n, name in enumerate(NAMES):
            with np.load(safe_file(source / name, models[name]['rows'][path]['map_path']), allow_pickle=False) as archive:
                predicted = archive['anomaly_map'] > anchor['models'][name]['threshold']
            counts = lookup[name][path]['pixel_counts']
            direct = {'TP': int((predicted & mask).sum()), 'FP': int((predicted & ~mask).sum()),
                      'FN': int((~predicted & mask).sum()), 'TN': int((~predicted & ~mask).sum())}
            if direct != counts:
                raise ValueError('Direct visual-mask counts differ from sorted counts')
            error = np.zeros((*mask.shape, 3), dtype=np.uint8)
            error[mask & predicted] = (0, 180, 0)
            error[~mask & predicted] = (255, 0, 0)
            error[mask & ~predicted] = (0, 100, 255)
            axes[n, 0].imshow(original)
            axes[n, 1].imshow(predicted, cmap='gray', vmin=0, vmax=1)
            axes[n, 2].imshow(error)
            for col in (0, 1):
                if mask.any():
                    axes[n, col].contour(mask, levels=[.5], colors=['yellow'], linewidths=.8)
            denom = counts['TP'] + counts['FP'] + counts['FN']
            iou = f'{counts["TP"] / denom:.3f}' if mask.any() else 'N/A (normal)'
            axes[n, 0].set_title(name + ': original / GT yellow', fontsize=10)
            axes[n, 1].set_title('Matched-recall mask; IoU=' + iou, fontsize=10)
            axes[n, 2].set_title(f'Green TP / red FP / blue FN\nFP={counts["FP"]}, FN={counts["FN"]}', fontsize=10)
        for axis in axes.flat:
            axis.axis('off')
        fig.suptitle(path + '\nOne GLOBAL cutoff per model, selected using development labels', fontsize=11)
        fig.tight_layout()
        destination = output / '可视化' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=110)
        entry = {'image_path': path, 'path': destination.relative_to(output).as_posix()}
        if mask.any() and float(mask.mean()) < .01:
            yy, xx = np.where(mask)
            for axis in axes.flat:
                axis.set_xlim(max(0, xx.min()-25), min(mask.shape[1], xx.max()+26))
                axis.set_ylim(min(mask.shape[0], yy.max()+26), max(0, yy.min()-25))
            zoom = destination.with_name(destination.stem + '_zoom.png')
            fig.savefig(zoom, dpi=110)
            entry['zoom_path'] = zoom.relative_to(output).as_posix()
        plt.close(fig)
        index.append(entry)
        if (number + 1) % 15 == 0:
            print(f'visual masks independently recounted: {number + 1}/{len(paths)}', flush=True)
    return index


def summaries(output, source, report, originals, anchor, selected, paired, curve):
    a, b = (flat(anchor, name) for name in NAMES)
    reduction = 1 - b['FP'] / a['FP']
    delta = b['macro_IoU'] - a['macro_IoU']
    lines = ['# 两种骨干网络的等召回率分析', '',
             '**开发集标签参与阈值选择；本报告是经验权衡分析，不是独立测试成绩或已验证的部署阈值。**', '',
             f'主对照目标固定为原ResNet18正常校准阈值下的缺陷像素召回率：{anchor["requested_recall"]:.8%}。',
             '每种网络只使用一个全局像素阈值，应用于全部45张开发图；不逐图调阈值。', '',
             '| 方案 | 实际像素召回 | 多圈正常像素FP | 平均IoU（37张） | 汇总IoU | 小面积IoU（3张） | 小面积召回 | 正常图误圈像素 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        r = flat(anchor, name)
        lines.append(f'| {name} | {r["actual_recall"]:.6%} | {r["FP"]:,} | {r["macro_IoU"]:.4f} | {r["pooled_IoU"]:.4f} | {r["small_IoU"]:.4f} | {r["small_recall"]:.4%} | {r["normal_FP"]:,} |')
    lines += ['', f'WideResNet50相对ResNet18：误圈像素变化{b["FP"]-a["FP"]:+,}（减少{reduction:.2%}），平均IoU变化{delta:+.4f}。',
              f'37张异常图的IoU变化：改善{paired["improved"]}张，下降{paired["worse"]}张，相同{paired["equal"]}张。',
              f'主对照两组实际召回差为{anchor["actual_recall_gap"]:.10f}；重复分数一起纳入，未随机拆分同分像素。', '',
              '## 预先固定的召回档位', '',
              '| 目标召回 | 网络 | 实际召回 | 误圈正常像素FP | 平均IoU | 小面积召回（n=3） |',
              '|---|---|---:|---:|---:|---:|']
    for p in selected:
        for name in NAMES:
            r = flat(p, name)
            lines.append(f'| {p["requested_recall"]:.2%} | {name} | {r["actual_recall"]:.6%} | {r["FP"]:,} | {r["macro_IoU"]:.4f} | {r["small_recall"]:.4%} |')
    lines += ['', '## 怎么理解和使用', '',
              '- 召回按全部真实缺陷像素汇总；大缺陷占权重更多，不保证每张图或小缺陷召回相同。小面积组与逐图指标均单列。',
              '- 误圈FP包括正常图及异常图中的正常区域；正常图误圈另列。平均IoU对37张异常图等权，汇总IoU对像素等权，两者不可混用。',
              '- 阈值在float32可表示值中取满足最低TP数的最大值；严格使用score > threshold。同分值导致轻微超出目标时公开实际召回与超出的TP数量。',
              '- 原始正常校准阈值仍保留。新阈值使用开发集缺陷标注，属于有监督的开发调参信息；不能称为正常数据独立校准，也不能直接作为最终泛化提升。',
              '- 本次只改像素掩膜的诊断阈值；整图分数、整图阈值和整图误报/漏检未调整。原0/0与1/0结果仍成立。',
              '- 重新阈值化不改变原始分数排序、AP或AUROC；AP差异沿用骨干对照，不能把AP变化归因于本分析。',
              '- 未训练、未推理、未重新测资源。WideResNet50此前处理耗时约2.43倍、PyTorch分配显存峰值约3.80倍的成本仍在。',
              '- 只有一个类别、一个种子，3张小面积图不足以证明普遍有效；开发集已多次使用。70张保留测试图未打开，当前模型及阈值配置未自动修改。',
              '- 若要采用新阈值，需先明确业务召回要求与是否允许缺陷标签校准，再冻结候选及阈值规则，最后按授权使用独立数据验证。', '',
              '本地报告提供45张等召回率对照和3张小面积放大图，所有逐图指标在matched_predictions.json与paired_localization.json。']
    text = '\n'.join(lines) + '\n'
    (output / '结果摘要.md').write_text(text, encoding='utf-8')
    public = {'status': 'completed_exploratory_development_matched_recall_not_final_test',
              'source_run': source.name, 'completed_at': now().isoformat(),
              'protocol': {'development_images': 45, 'anomalous_images': 37, 'normal_images': 8,
                           'small_area_images': 3, 'held_out_images_opened': 0, 'new_inference': False,
                           'global_pixel_threshold_per_model': True, 'decision': 'score > float32 threshold',
                           'threshold_selection_uses_development_defect_labels': True,
                           'anchor': 'original normally calibrated ResNet18 pooled pixel recall',
                           'image_decisions_unchanged': True, 'automatic_adoption': False},
              'anchor': {k: v for k, v in anchor.items() if k != 'models'},
              'anchor_models': {name: flat(anchor, name) for name in NAMES},
              'prespecified_points': [{k: v for k, v in p.items() if k != 'models'} | {'models': {name: flat(p, name) for name in NAMES}} for p in selected],
              'paired_IoU_changes': paired,
              'reference_normally_calibrated_thresholds': report['thresholds'],
              'original_quality': {name: {'pixel_counts': originals[name]['pixel_counts'],
                                        'macro_IoU': originals[name]['anomaly_macro']['pixel_IoU']} for name in NAMES},
              'FP_reduction_fraction_at_anchor': reduction, 'macro_IoU_delta_at_anchor': delta,
              'maximum_actual_recall_gap_on_curve': max(p['actual_recall_gap'] for p in curve)}
    write(output / 'public_summary.json', public)
    return text, public


def html_summary(text):
    """Render the small fixed report format without external dependencies."""
    result, table = [], False
    for line in text.splitlines():
        if line.startswith('|'):
            cells = [part.strip() for part in line.strip('|').split('|')]
            if all(set(c) <= set('-:') for c in cells):
                continue
            if not table:
                result.append('<div style="overflow:auto"><table>')
                table = True
                tag = 'th'
            else:
                tag = 'td'
            result.append('<tr>' + ''.join(f'<{tag}>' + html.escape(c) + f'</{tag}>' for c in cells) + '</tr>')
        else:
            if table:
                result.append('</table></div>')
                table = False
            if line.startswith('# '):
                continue
            if line.startswith('## '):
                result.append('<h2>' + html.escape(line[3:]) + '</h2>')
            elif line:
                result.append('<p>' + html.escape(line.strip('*')) + '</p>')
    if table:
        result.append('</table></div>')
    return ''.join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--data-root', type=Path, default=DATA)
    args = parser.parse_args()
    for path in (args.source, args.data_root):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('All project inputs must remain within the canonical project root')
    output = ROOT / '04实验结果/10等召回率分析' / ('matched_recall_' + now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'status.json', {'status': 'running', 'started_at': now().isoformat(), 'source_run': args.source.name})
    write(output / 'protocol.json', {'source_run': args.source.name,
          'fixed_recall_targets': [.90, .95, .97, .99],
          'primary_target': 'frozen_resnet18_normally_calibrated_pixel_recall',
          'descriptive_grid': '90% through 99% in 0.1 percentage-point steps',
          'global_cutoff_per_model': True, 'float32_strict_greater_than': True,
          'development_labels_used_for_cutoff_selection': True, 'held_out_images_to_open': 0})
    manifest, report, models, masks, identities, hashes, originals = load_inputs(args.source, args.data_root)
    paths = manifest['splits']['development']
    counts = originals['resnet18']['pixel_counts']
    total_pos = counts['TP'] + counts['FN']
    anchor, rows = point(models, paths, counts['TP'], total_pos, 'frozen_resnet18_recall')
    selected = [point(models, paths, int((Decimal(str(r)) * total_pos).to_integral_value(rounding=ROUND_CEILING)),
                      total_pos, f'recall_{r}')[0] for r in (.90, .95, .97, .99)]
    # A predeclared descriptive grid, not a search for the best result.
    curve = [point(models, paths, int((Decimal(900 + i) * total_pos / 1000).to_integral_value(rounding=ROUND_CEILING)),
                   total_pos, f'grid_{900+i}')[0] for i in range(91)]
    write(output / 'anchor.json', anchor)
    write(output / 'prespecified_points.json', selected)
    write(output / 'recall_curve.json', curve)
    write(output / 'matched_predictions.json', rows)
    paired_rows, tally = [], {'improved': 0, 'worse': 0, 'equal': 0}
    lookup = {name: {r['image_path']: r for r in rows[name]} for name in NAMES}
    for path in paths:
        entry = {'image_path': path, 'label': manifest['files'][path]['label'], 'models': {}}
        for name in NAMES:
            c = lookup[name][path]['pixel_counts']
            d = c['TP'] + c['FP'] + c['FN']
            entry['models'][name] = {'counts': c, 'IoU': c['TP'] / d if entry['label'] else None,
                                    'recall': c['TP'] / (c['TP']+c['FN']) if entry['label'] else None}
        if entry['label']:
            delta = entry['models'][NAMES[1]]['IoU'] - entry['models'][NAMES[0]]['IoU']
            entry['IoU_delta'] = delta
            tally['improved' if delta > 0 else 'worse' if delta < 0 else 'equal'] += 1
        paired_rows.append(entry)
    write(output / 'paired_localization.json', paired_rows)
    draw_curve(output, curve, anchor)
    images = visualize(output, args.source, args.data_root, paths, models, masks, anchor, rows)
    write(output / 'visualization_index.json', images)
    text, public = summaries(output, args.source, report, originals, anchor, selected, tally, curve)
    details = ''.join('<details><summary>' + html.escape(r['image_path']) + '</summary><img loading="lazy" src="' + html.escape(r['path']) + '">' +
                      ('<p>小面积缺陷放大图</p><img loading="lazy" src="' + html.escape(r['zoom_path']) + '">' if 'zoom_path' in r else '') + '</details>' for r in images)
    (output / '对照报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>等召回率分析</title><style>body{max-width:1400px;margin:30px auto;padding:20px;font:16px/1.6 sans-serif}img{width:100%}summary{cursor:pointer;padding:12px}a{color:#1675b8}table{border-collapse:collapse;font-size:15px}td,th{padding:10px;border:1px solid #ddd;text-align:left}th{background:#edf3f8}</style><h1>等召回率分析：误圈面积与定位质量</h1><p><a href="结果摘要.md">查看Markdown指标表</a> · <a href="public_summary.json">机器可读摘要</a></p>' + html_summary(text) + '<img src="召回与误圈权衡.png"><p>以下全部45张图采用主对照的每网络全局阈值。上行ResNet18，下行WideResNet50；绿=正确覆盖、红=误圈、蓝=漏圈，黄=人工标注边界。</p>' + details + '</html>', encoding='utf-8')
    for relative, digest in identities.items():
        if file_sha256(args.source / relative) != digest:
            raise ValueError('Frozen source changed during analysis')
    for relative, digest in hashes.items():
        if file_sha256(args.source / relative) != digest:
            raise ValueError('Frozen map changed during analysis')
    write(output / 'verification.json', {'source_metadata_sha256': identities, 'consumed_development_maps_sha256': hashes,
          'manifest_sha256': manifest['manifest_sha256'], 'source_evidence_unchanged': True,
          'frozen_counts_and_quality_reproduced': True, 'direct_mask_counts_checked': 90,
          'image_decisions_unchanged': True, 'held_out_images_opened': 0,
          'source_sha256': {f: file_sha256(Path(__file__).with_name(f)) for f in ('analyze_matched_recall.py', 'matched_recall_metrics.py')},
          'numpy_version': np.__version__})
    write(output / 'status.json', {'status': 'completed', 'completed_at': now().isoformat(), 'new_inference': False})
    print(json.dumps(public['anchor_models'], ensure_ascii=False, indent=2), flush=True)
    print('MATCHED_RECALL_COMPLETED | ' + str(output), flush=True)


if __name__ == '__main__':
    main()
