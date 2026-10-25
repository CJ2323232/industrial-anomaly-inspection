"""Exact empirical global-IoU optimization with declared recall constraints.

This is development-label-informed selection, not deployment calibration. No
model is instantiated. Examines all positive-score levels, not a coarse grid.
"""
import argparse
from pathlib import Path
import json

import numpy as np
from PIL import Image

if __package__:
    from .analyze_matched_recall import ROOT, SOURCE, DATA, read, write, html_summary
    from .baseline_utils import load_manifest, safe_file, file_sha256
    from .matched_recall_metrics import sorted_scores
    from .run_threshold_scan import evaluate
    from .threshold_scan_reporting import metrics_table
else:
    from analyze_matched_recall import ROOT, SOURCE, DATA, read, write, html_summary
    from baseline_utils import load_manifest, safe_file, file_sha256
    from matched_recall_metrics import sorted_scores
    from run_threshold_scan import evaluate
    from threshold_scan_reporting import metrics_table


def best_constrained_cutoff(positive_sorted, negative_sorted, minimum_tp, maximum_threshold):
    levels, counts = np.unique(positive_sorted, return_counts=True)
    thresholds = np.nextafter(levels, np.float32(-np.inf))
    tp = np.cumsum(counts[::-1], dtype=np.int64)[::-1]
    fp = len(negative_sorted) - np.searchsorted(negative_sorted, levels, side='left')
    allowed = (tp >= minimum_tp) & (thresholds <= maximum_threshold)
    if not allowed.any():
        raise ValueError('No feasible threshold')
    iou = tp / (len(positive_sorted) + fp)
    best = np.flatnonzero(allowed & (iou == np.max(iou[allowed])))[-1]
    return {'threshold': float(thresholds[best]), 'Pixel_IoU': float(iou[best]),
            'positive_score_levels_examined': len(levels), 'feasible_levels': int(allowed.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    output = args.analysis.resolve()
    if not output.is_relative_to(ROOT) or (output / 'recommended_plan.json').exists():
        raise ValueError('Require canonical analysis folder and new plan output')
    report = read(output / 'public_summary.json')
    inventory, labels = report['inventory'], report['labels']
    manifest = load_manifest(SOURCE / 'split_manifest.json')
    paths = manifest['splits']['development']
    all_rows = read(output / 'matched_working_points.json')
    minimum_tp = all_rows['resnet18']['baseline_recall']['required_TP']
    masks = {}
    for path in paths:
        record = manifest['files'][path]
        if record['mask_path']:
            file = safe_file(DATA, record['mask_path'])
            if file_sha256(file) != record['mask_sha256']:
                raise ValueError('GT identity mismatch')
            with Image.open(file) as image:
                masks[path] = np.asarray(image) > 0
        else:
            masks[path] = np.zeros((record['height'], record['width']), dtype=bool)
    plans, thresholds, per_images = {}, {}, {}
    for name, inv in inventory.items():
        directory = ROOT / inv['source']
        rows = {r['image_path']: r for r in all_rows[name]['baseline_recall']['per_image']}
        values = {}
        for path in paths:
            with np.load(safe_file(directory, 'maps/development/' + Path(path).with_suffix('.npz').as_posix()), allow_pickle=False) as z:
                array = z['anomaly_map']
            mask = masks[path]
            values[path] = sorted_scores(array[mask]), sorted_scores(array[~mask])
        pos = np.sort(np.concatenate([v[0] for v in values.values()]))
        neg = np.sort(np.concatenate([v[1] for v in values.values()]))
        small_min = min(float(values[path][0][0]) for path in paths if rows[path]['label'] and rows[path]['area_ratio'] < .01)
        maximum_threshold = float(np.nextafter(np.float32(small_min), np.float32(-np.inf)))
        optimum = best_constrained_cutoff(pos, neg, minimum_tp, maximum_threshold)
        summary, individual = evaluate({'rows': rows, 'scores': values}, paths, optimum['threshold'])
        assert summary['small_pixel_recall'] == 1 and summary['TP'] >= minimum_tp
        assert summary['Pixel_IoU'] == optimum['Pixel_IoU']
        plans[name], thresholds[name], per_images[name] = summary, optimum, individual
        print(name, json.dumps(optimum), flush=True)
    winner = max(plans, key=lambda n: plans[n]['Pixel_IoU'])
    baseline = report['matched_anchor']['resnet18']
    selected = plans[winner]
    pair = list(zip(per_images['resnet18'], per_images[winner]))
    changes = {'improved': sum(a['label'] and b['Pixel_IoU'] > a['Pixel_IoU'] for a, b in pair),
               'worse': sum(a['label'] and b['Pixel_IoU'] < a['Pixel_IoU'] for a, b in pair)}
    plan = {'selection_rule': 'Exact maximum pooled Pixel IoU among5 stored configurations and all global thresholds, subject to pooled Recall>=frozen baseline and100% recall on3 known small-area images',
            'uses_development_defect_GT': True, 'recommended_method': winner, 'recommended_summary': selected,
            'candidate_optima': plans, 'exact_search': thresholds, 'per_image': per_images,
            'IoU_change_counts_vs_constrained_resnet18': changes,
            'minimum_TP': minimum_tp, 'FP_reduction_vs_original_matched_baseline': 1-selected['FP']/baseline['FP'],
            'formal_deployment_calibration': False, 'automatic_adoption': False, 'held_out_images_opened': 0}
    write(output / 'recommended_plan.json', plan)
    lines = ['# 当前项目的明确方案推荐', '',
             f'建议把 **{labels[winner]}，全局像素阈值{selected["threshold"]:.10f}** 作为下一阶段定位验证的主候选。整图阈值沿用该模型原规则，不在本次改动。', '',
             '选择依据明确：总体缺陷像素召回至少达到原基线95.904%，且开发集3张小面积图的缺陷像素全部覆盖；在满足这两项条件的所有全局阈值和5种现有配置里，选汇总Pixel IoU最高者。',
             '搜索逐一检查每个真实缺陷分数层级，重复分数一起处理。只增加正常像素、不增加TP的分数层级不会提高IoU，因此这是上述约束下的精确经验最优，不是仅取501点网格中的最优。', '',
             *metrics_table(plans, labels), '',
             f'主推荐相对原基线：总体召回{baseline["Pixel_Recall"]:.4%}→{selected["Pixel_Recall"]:.4%}；汇总Pixel IoU {baseline["Pixel_IoU"]:.4f}→{selected["Pixel_IoU"]:.4f}；Precision {baseline["Pixel_Precision"]:.4%}→{selected["Pixel_Precision"]:.4%}；FP {baseline["FP"]:,}→{selected["FP"]:,}（减少{plan["FP_reduction_vs_original_matched_baseline"]:.2%}）；小面积召回100%→100%。', '',
             '## 为什么不直接选2.127015', '',
             '2.127015是新骨干在严格等总体召回对照中的工作点：汇总IoU0.6941、平均IoU0.5013、FP985275，但小面积组漏64个像素（color/001漏63、bent/002漏1）。本推荐增加“小缺陷全部覆盖”的条件，允许更宽一些的掩膜，换取这3张已知小缺陷的完整覆盖；没有声称未知小缺陷也一定不漏。', '',
             '## 每个指标怎样理解', '',
             '- Recall：真实缺陷里找到了多少；95.9%意味着约4.1%的缺陷像素漏圈。',
             '- IoU：圈出区域与真实缺陷的交并比；同时惩罚多圈和漏圈。这里主指标把全部45张图的像素计数先累加。',
             '- Precision：圈出的区域里有多少是真的缺陷；其余为误圈。',
             '- FP：误圈的正常像素总数。误圈面积占比=FP/全部45张图的像素数，不是FP/正常像素数。',
             '- 平均IoU：对37张异常图等权平均；用于检查总体像素指标是否被大缺陷主导。', '',
             '## 必须接受的具体代价', '',
             '- WideResNet50原实测处理时间约为同次ResNet18的2.43倍（7.015→17.047ms），PyTorch分配显存峰值38.51→146.45MiB。调像素阈值不会消除这项成本。',
             '- 原整图误报仍为1/8（good/005），基线为0/8；两者整图漏检均0/37。本次优化的是定位掩膜，不能写成整图误报也改善。',
             '- 个别缺陷边界仍退步；主扫描报告提供所有逐图结果及退步案例。新骨干14/37张AP下降，8/37张即使逐图选择最佳阈值，IoU仍不到0.5，说明单调阈值不是全部问题。', '',
             '## 如何使用这项选择', '',
             '将它作为一个待独立验证的开发候选，保留原基线作对照。这里的最优使用了开发GT，既不是正式部署校准，也不是最终泛化结论。网络、模型权重和现有配置均未修改，70张保留测试图仍未访问。',
             '如果目标只追求同总体召回下更紧的掩膜，主扫描表中的2.127015更合适；本项目强调小缺陷覆盖，因此主推荐采用上面的约束最优工作点。']
    text = '\n'.join(lines) + '\n'
    (output / '方案选择说明.md').write_text(text, encoding='utf-8')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>项目方案推荐</title><style>body{max-width:1400px;margin:30px auto;padding:20px;font:16px/1.7 "Microsoft YaHei",sans-serif;color:#233749}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ddd;padding:10px}th{background:#edf3f8}a{color:#176ca6}</style><h1>项目方案推荐与取舍</h1><p><a href="对照报告.html">完整阈值扫描、曲线与全部逐图指标</a> · <a href="recommended_plan.json">选择依据与完整数值</a></p>' + html_summary(text) + '</html>'
    (output / '方案选择报告.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
