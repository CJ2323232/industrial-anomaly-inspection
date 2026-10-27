"""Audit saved development maps and exact per-image threshold bounds; no inference.

All paths are manifest-allowlisted. Oracle masks use development GT and are
diagnostic upper bounds, never deployable outputs. Existing reports are read only.
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

if __package__:
    from .analyze_matched_recall import ROOT, SOURCE, DATA, read, write, now, html_summary
    from .baseline_utils import safe_file, file_sha256, load_manifest
    from .threshold_scan_metrics import pixel_metrics, oracle_image_iou
else:
    from analyze_matched_recall import ROOT, SOURCE, DATA, read, write, now, html_summary
    from baseline_utils import safe_file, file_sha256, load_manifest
    from threshold_scan_metrics import pixel_metrics, oracle_image_iou

SHORT = {'resnet18': 'R18-256', 'wide_resnet50': 'WRN50-256',
         'resnet18_512': 'R18-512', 'resnet18_sigma2': 'R18-sigma2',
         'resnet18_tiled': 'R18-overlap'}


def measure(scores, mask, threshold):
    pred = scores > threshold
    return pixel_metrics({'TP': int((pred & mask).sum()), 'FP': int((pred & ~mask).sum()),
                          'FN': int((~pred & mask).sum()), 'TN': int((~pred & ~mask).sum())})


def error_rgb(pred, mask):
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    rgb[pred & mask] = (0, 185, 95)
    rgb[pred & ~mask] = (240, 70, 65)
    rgb[~pred & mask] = (50, 125, 255)
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    source = args.analysis.resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError('Source must be inside canonical project')
    names = ['ranking_metrics.json', 'recommended_plan.json', 'matched_working_points.json',
             'original_working_points.json', 'data_inventory.json', 'public_summary.json']
    hashes = {n: file_sha256(source / n) for n in names}
    ranking, plan, matched, original, inventory, public = [read(source / n) for n in names]
    manifest = load_manifest(SOURCE / 'split_manifest.json')
    paths = manifest['splits']['development']
    assert len(paths) == 45 and len(set(paths)) == 45
    rows = {n: {r['image_path']: r for r in matched[n]['baseline_recall']['per_image']} for n in inventory}
    assert all(set(rs) == set(paths) for rs in rows.values())
    anomaly_paths = [p for p in paths if rows['resnet18'][p]['label']]
    assert len(anomaly_paths) == 37
    hard = sorted(p for p in anomaly_paths if ranking['wide_resnet50']['per_image'][p]['oracle']['max_IoU'] < .5)
    output = ROOT / '04实验结果/12定位失败诊断' / ('failure_audit_' + now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    (output / '案例图').mkdir()
    all_cases, inputs, checked = [], [], 0
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for path in anomaly_paths:
        record = manifest['files'][path]
        mask_file = safe_file(DATA, record['mask_path'])
        assert file_sha256(mask_file) == record['mask_sha256']
        with Image.open(mask_file) as im:
            mask = np.asarray(im) > 0
        distance = distance_transform_edt(~mask)
        maps, details = {}, {}
        for name, inv in inventory.items():
            map_file = safe_file(ROOT / inv['source'], 'maps/development/' + Path(path).with_suffix('.npz').as_posix())
            with np.load(map_file, allow_pickle=False) as z:
                score = z['anomaly_map']
            assert score.dtype == np.float32 and score.shape == mask.shape and np.isfinite(score).all()
            maps[name] = score
            inputs.append({'path': map_file.relative_to(ROOT).as_posix(), 'sha256': file_sha256(map_file)})
            oracle = oracle_image_iou(np.sort(score[mask]), np.sort(score[~mask]))
            prior = ranking[name]['per_image'][path]['oracle']
            assert oracle == prior, (name, path, oracle, prior)
            metrics = measure(score, mask, oracle['threshold'])
            assert abs(metrics['Pixel_IoU'] - oracle['max_IoU']) < 1e-12
            global_threshold = matched[name]['baseline_recall']['summary']['threshold']
            gm = measure(score, mask, global_threshold)
            assert all(gm[k] == rows[name][path][k] for k in ['TP', 'FP', 'FN', 'TN'])
            fp_mask = (score > global_threshold) & ~mask
            near = {str(d): int((fp_mask & (distance <= d)).sum()) for d in (5, 10, 20)}
            details[name] = {'AP': ranking[name]['per_image'][path]['AP'],
                'matched_global': gm, 'global_threshold': global_threshold,
                'oracle': metrics | {'threshold': oracle['threshold']},
                'oracle_minus_global_IoU': oracle['max_IoU'] - gm['Pixel_IoU'],
                'global_FP_within_GT_distance_pixels': near,
                'global_FP_beyond_20px': gm['FP'] - near['20']}
            checked += 1
        best = max(details, key=lambda n: details[n]['oracle']['Pixel_IoU'])
        wrn = details['wide_resnet50']
        case = {'image_path': path, 'defect_type': rows['resnet18'][path]['defect_type'],
                'area_ratio': rows['resnet18'][path]['area_ratio'], 'methods': details,
                'WRN_oracle_below_half': path in hard, 'best_oracle_method': best,
                'all_methods_oracle_below_half': max(d['oracle']['Pixel_IoU'] for d in details.values()) < .5}
        if path in hard:
            image_file = safe_file(DATA, path)
            assert file_sha256(image_file) == record['image_sha256']
            with Image.open(image_file) as im:
                image = np.asarray(im)
            fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
            axes[0].imshow(image); axes[0].contour(mask, [.5], colors=['yellow'], linewidths=.8)
            axes[0].set_title('Original / GT yellow')
            for ax, name, key in [(axes[1], 'wide_resnet50', 'matched_global'),
                                  (axes[2], 'wide_resnet50', 'oracle'), (axes[3], best, 'oracle')]:
                d = details[name]; m = d[key]
                t = d['global_threshold'] if key == 'matched_global' else m['threshold']
                ax.imshow(error_rgb(maps[name] > t, mask))
                ax.set_title(f'{SHORT[name]} / {"global" if key == "matched_global" else "GT oracle"}\nIoU {m["Pixel_IoU"]:.3f}, Recall {m["Pixel_Recall"]:.3f}\nFP {m["FP"]}, FN {m["FN"]}')
            for ax in axes:
                ax.axis('off')
            fig.suptitle(path + '\nGreen=TP, red=FP, blue=FN. GT oracle uses an individual threshold; not a deployable result.')
            fig.tight_layout(rect=(0, 0, 1, .84))
            filename = path.replace('/', '_')
            fig.savefig(output / '案例图' / filename, dpi=120); plt.close(fig)
            case['figure'] = '案例图/' + filename
        all_cases.append(case)
        print('checked', len(all_cases), '/', len(anomaly_paths), path, flush=True)

    summary = {}
    for name in inventory:
        ds = [c['methods'][name] for c in all_cases]
        summary[name] = {'label': inventory[name]['label'],
            'matched_global_macro_IoU': float(np.mean([d['matched_global']['Pixel_IoU'] for d in ds])),
            'oracle_macro_IoU': float(np.mean([d['oracle']['Pixel_IoU'] for d in ds])),
            'oracle_below_half': sum(d['oracle']['Pixel_IoU'] < .5 for d in ds),
            'oracle_minus_global_macro_IoU': float(np.mean([d['oracle_minus_global_IoU'] for d in ds])),
            'oracle_threshold_min': min(d['oracle']['threshold'] for d in ds),
            'oracle_threshold_max': max(d['oracle']['threshold'] for d in ds),
            'matched_global_FP_total': sum(d['matched_global']['FP'] for d in ds),
            'matched_global_FP_beyond_20px': sum(d['global_FP_beyond_20px'] for d in ds)}
    hard_rows = [c for c in all_cases if c['WRN_oracle_below_half']]
    stats = {'WRN_hard_cases': len(hard_rows),
        'hard_cases_512_oracle_above_half': sum(c['methods']['resnet18_512']['oracle']['Pixel_IoU'] >= .5 for c in hard_rows),
        'hard_cases_any_other_oracle_above_half': sum(not c['all_methods_oracle_below_half'] for c in hard_rows),
        'all_methods_oracle_below_half_count': sum(c['all_methods_oracle_below_half'] for c in all_cases),
        'WRN_oracle_better_than_baseline_images': sum(c['methods']['wide_resnet50']['oracle']['Pixel_IoU'] > c['methods']['resnet18']['oracle']['Pixel_IoU'] for c in all_cases)}
    write(output / '逐图诊断.json', all_cases)
    write(output / '汇总.json', {'methods': summary, 'findings': stats})
    write(output / '审计记录.json', {'source': source.relative_to(ROOT).as_posix(), 'source_hashes': hashes,
        'input_map_hashes': inputs, 'verified_maps': checked, 'verified_oracle_thresholds': checked,
        'verified_matched_thresholds': checked, 'development_images': 45, 'anomaly_images_analyzed': 37,
        'normal_images_metrics_reused_from_prior_45_image_analysis': 8,
        'held_out_images_opened': 0, 'model_forward_passes': 0, 'training_runs': 0,
        'oracle_uses_development_GT': True, 'FP_distance_unit': 'original 700x700 image pixel',
        'source_files_unchanged': all(file_sha256(source / n) == h for n, h in hashes.items())})

    lines = ['# 定位失败诊断：下一步先解决什么', '',
        '当前保留 R18/256/sigma4 为成本基线，WRN50/256/sigma4 为高成本定位候选。此前推荐阈值2.103178仅作为开发候选，未写入模型配置。', '',
        '本次检查的是同总体召回工作点与逐图最优阈值上界之间的差距。逐图上界允许每张图片看GT后分别选阈值，不能部署，也不能当作最终性能。', '',
        '|方案|同总体召回下逐图平均IoU|逐图阈值上界平均IoU|差距|上界仍低于0.5的张数|',
        '|---|---:|---:|---:|---:|']
    for n,s in summary.items():
        lines.append(f'|{s["label"]}|{s["matched_global_macro_IoU"]:.4f}|{s["oracle_macro_IoU"]:.4f}|{s["oracle_minus_global_macro_IoU"]:.4f}|{s["oracle_below_half"]}/37|')
    lines += ['', '差距代表在已保存分数图上改变逐图阈值的乐观空间，不代表真实可获得的提升；它还可能伴随召回下降。上界不受原来的召回约束，因此不能与高召回方案直接宣布胜负。', '',
        '## 8张新网络难例', '', '|图片|WRN同召回IoU|WRN上界IoU|R18-512上界IoU|五方案中上界最好|', '|---|---:|---:|---:|---|']
    for c in hard_rows:
        d=c['methods']; best=c['best_oracle_method']
        lines.append(f'|{c["image_path"]}|{d["wide_resnet50"]["matched_global"]["Pixel_IoU"]:.4f}|{d["wide_resnet50"]["oracle"]["Pixel_IoU"]:.4f}|{d["resnet18_512"]["oracle"]["Pixel_IoU"]:.4f}|{SHORT[best]}：{d[best]["oracle"]["Pixel_IoU"]:.4f}|')
    lines += ['', f'8张难例中，512的逐图上界有{stats["hard_cases_512_oracle_above_half"]}张达到0.5；五种方案全部都无法达到0.5的难例共{stats["all_methods_oracle_below_half_count"]}张。', '',
        '## 明确的下一步', '',
        '1. 暂不加大网络、暂不继续重叠分块。先验证分数校准能否把512已有的逐图潜力转化为可共用的全局阈值。这个优先级来自本次差距诊断，不是收益保证。',
        '2. 先做开发集内部交叉验证的阈值稳定性实验：按图片和缺陷类型划分5折，在其余折上选择全局阈值，仅评估未参与选阈值的一折；整图划分，禁止像素随机划分。用预先固定的Recall目标95%、97%、99%，报告留出折实际Recall、IoU、FP及逐图值。每张图只作为验证一次，汇总45张预测。不得称为独立最终测试，因为这些图片已参与前期探索。',
        '3. 如果跨折阈值和实际召回明显波动，再准备基于正常校准集的分数标准化方案；这需要另行扩大当前仅开发集的诊断范围，本次没有读取44张正常校准图或它们的分数图。',
        '4. 对所有配置逐图阈值上界仍不足的图片，保留为定位能力难例。后续再讨论局部特征或边界细化；不把调阈值描述为已经解决它们。', '',
        '## 指标和图例', '',
        '- 同总体召回：全45张图累计TP/缺陷像素约95.904%，不保证每张图召回相同。',
        '- 平均IoU：37张异常图各占同等权重，交集/并集；这里不是汇总Pixel IoU。',
        '- 逐图上界：每张图片在全部真实分数阈值上寻找最大IoU，没有模型训练；它使用GT，是乐观诊断上界。',
        '- 红色=正常像素被多圈FP；蓝色=缺陷像素未圈到FN；绿色=正确圈到TP。',
        '- 距离统计：原始700×700坐标中，误圈像素到最近GT缺陷像素的欧氏距离。5/10/20像素是描述性距离带，不能仅凭远近断言由反光或平滑引起。', '',
        '## 边界', '',
        '没有重新训练、运行模型前向或修改网络；没有访问70张保留测试图片；没有自动替换当前基线。已核对185张连续分数图及其GT，重算185个逐图上界并核对185个共用阈值的像素计数，原报告保持不变。全部结论属于当前开发集诊断。']
    text='\n'.join(lines)+'\n'
    (output/'诊断与下一步.md').write_text(text, encoding='utf-8')

    cards=''.join(f'<section><h3>{html.escape(c["image_path"])}</h3><img loading="lazy" src="{html.escape(c["figure"])}"></section>' for c in hard_rows)
    table='<table><thead><tr><th>图片</th><th>配置</th><th>全局IoU</th><th>全局Recall</th><th>FP</th><th>FP距GT&gt;20px</th><th>上界IoU</th><th>上界Recall</th></tr></thead><tbody>'
    for c in all_cases:
        for n,d in c['methods'].items():
            g,o=d['matched_global'],d['oracle']
            table+=f'<tr><td>{c["image_path"]}</td><td>{SHORT[n]}</td><td>{g["Pixel_IoU"]:.4f}</td><td>{g["Pixel_Recall"]:.2%}</td><td>{g["FP"]}</td><td>{d["global_FP_beyond_20px"]}</td><td>{o["Pixel_IoU"]:.4f}</td><td>{o["Pixel_Recall"]:.2%}</td></tr>'
    table+='</tbody></table>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>定位失败诊断</title><style>body{max-width:1250px;margin:35px auto;padding:0 22px;font:16px/1.7 system-ui;color:#182c40;background:#f5f7fa}h1,h2{color:#125475}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:9px;border:1px solid #dbe3ec;text-align:left}th{background:#e4eef7}img{width:100%;background:white}section{margin:30px 0}details{margin:25px 0}p,li{max-width:1050px}</style>'+'<h1>定位失败诊断：下一步先解决什么</h1>'+html_summary(text)+cards+'<details><summary>展开全部37张 × 5种方案的逐图数据</summary>'+table+'</details></html>'
    (output/'定位失败诊断.html').write_text(page,encoding='utf-8')
    print(json.dumps({'output': str(output), 'findings':stats, 'methods':summary}, ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
