"""Standalone figures and Chinese report for saved-map threshold diagnostics."""
import html
import json
from pathlib import Path

import numpy as np

if __package__:
    from .analyze_matched_recall import html_summary, write
    from .baseline_utils import safe_file
else:
    from analyze_matched_recall import html_summary, write
    from baseline_utils import safe_file

COLORS = ['#2075b5', '#e17718', '#389866', '#9869ab', '#cc5760']
SHORT = {'resnet18': 'R18 /256 /s4', 'wide_resnet50': 'WRN50 /256 /s4',
         'resnet18_512': 'R18 /512 /s4', 'resnet18_sigma2': 'R18 /256 /s2', 'resnet18_tiled': 'R18 /overlap256'}


def number(value, percent=False, digits=4):
    return '未定义' if value is None else (f'{value:.{digits}%}' if percent else f'{value:.{digits}f}')


def metrics_table(values, labels):
    lines = ['| 方案 | Pixel Recall | Pixel IoU（汇总） | Pixel Precision | FP像素 | 误圈面积占全图 | 平均IoU（37图） | 小面积召回（3图） |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in values.items():
        lines.append(f'| {labels[name]} | {number(row["Pixel_Recall"], True)} | {row["Pixel_IoU"]:.4f} | {number(row["Pixel_Precision"], True)} | {row["FP"]:,} | {number(row["FP_area_fraction"], True)} | {row["macro_IoU_37_anomalies"]:.4f} | {number(row["small_pixel_recall"], True)} |')
    return lines


def figures(output, scans, matched, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric, ylabel, filename in [('Pixel_IoU', 'Pixel IoU (pooled across all45 development images)', 'IoU-Recall.png'),
                                      ('FP', 'FP pixels (normal pixels marked defective)', 'FP-Recall.png')]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for (name, points), color in zip(scans.items(), COLORS):
            for axis in axes:
                axis.plot([p['Pixel_Recall'] * 100 for p in points], [p[metric] for p in points], label=SHORT[name], color=color)
                anchor = matched[name]['baseline_recall']['summary']
                axis.scatter(anchor['Pixel_Recall'] * 100, anchor[metric], color=color, marker='*', s=80, zorder=4)
                axis.grid(alpha=.2)
                axis.set(xlabel='Pooled defect-pixel recall (%)', ylabel=ylabel)
            axes[0].set_xlim(0, 100)
            axes[1].set_xlim(90, 99)
        axes[0].set_title('Full common reachable interval:0..100%')
        axes[1].set_title('High-recall detail:90..99%; stars=baseline recall target')
        if metric == 'FP':
            # Avoid a full-range maximum flattening the high-recall comparison.
            high = [p[metric] for points in scans.values() for p in points if .9 <= p['Pixel_Recall'] <= .99001]
            axes[1].set_ylim(0, max(high) * 1.05)
        else:
            high = [p[metric] for points in scans.values() for p in points if .9 <= p['Pixel_Recall'] <= .99001]
            axes[1].set_ylim(max(0, min(high) - .03), min(1, max(high) + .03))
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        fig.suptitle('Saved-score threshold diagnostics on development labels; no deployment calibration')
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (name, points), color in zip(scans.items(), COLORS):
        axes[0].plot([p['Pixel_Recall'] * 100 for p in points], [p['Pixel_Precision'] for p in points], label=SHORT[name], color=color)
        axes[1].plot([p['Pixel_Recall'] * 100 for p in points], [p['FP_area_fraction'] * 100 for p in points], label=SHORT[name], color=color)
    for axis in axes:
        axis.set_xlim(90, 99); axis.grid(alpha=.2); axis.legend(fontsize=8)
        axis.set_xlabel('Pooled defect-pixel recall (%)')
    axes[0].set_ylabel('Pixel precision')
    axes[1].set_ylabel('FP / all image pixels (%)')
    fig.tight_layout(); fig.savefig(output / 'Precision与误圈占比.png', dpi=150); plt.close(fig)


def case_images(output, baseline, winner, models, directories, masks, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    paired = []
    left = {r['image_path']: r for r in baseline['per_image']}
    right = {r['image_path']: r for r in winner['per_image']}
    for path, row in left.items():
        if row['label']:
            paired.append({'image_path': path, 'IoU_delta': right[path]['Pixel_IoU'] - row['Pixel_IoU'],
                           'baseline': row, 'candidate': right[path]})
    sorted_rows = sorted(paired, key=lambda r: r['IoU_delta'])
    selected = {r['image_path'] for r in sorted_rows[:3] + sorted_rows[-3:]}
    selected.update(path for path, row in left.items() if row['label'] and row['area_ratio'] < .01)
    winner_name = winner['method']
    images = []
    for path in sorted(selected):
        with Image.open(safe_file(data, path)) as im:
            original = np.asarray(im)
        mask = masks[path]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for n, (method, point) in enumerate([('resnet18', baseline), (winner_name, winner)]):
            with np.load(safe_file(directories[method], models[method]['rows'][path]['map_path']), allow_pickle=False) as z:
                predicted = z['anomaly_map'] > point['summary']['threshold']
            axes[n, 0].imshow(original)
            axes[n, 1].imshow(predicted, cmap='gray', vmin=0, vmax=1)
            error = np.zeros((*mask.shape, 3), dtype=np.uint8)
            error[predicted & mask] = (0, 180, 0); error[predicted & ~mask] = (255, 0, 0); error[~predicted & mask] = (0, 100, 255)
            axes[n, 2].imshow(error)
            for col in (0, 1):
                axes[n, col].contour(mask, levels=[.5], colors=['yellow'], linewidths=.8)
            axes[n, 0].set_title(SHORT[method] + ': original / GT yellow')
            row = left[path] if n == 0 else right[path]
            axes[n, 1].set_title(f'IoU={row["Pixel_IoU"]:.3f}; recall={row["Pixel_Recall"]:.3f}')
            axes[n, 2].set_title(f'Green TP / red FP / blue FN\nFP={row["FP"]}, FN={row["FN"]}')
        for axis in axes.flat:
            axis.axis('off')
        fig.suptitle(path + '\nGlobal thresholds selected at the baseline pooled recall; not per-image optimization')
        fig.tight_layout()
        dest = output / '案例图' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=110)
        plt.close(fig)
        images.append({'image_path': path, 'path': dest.relative_to(output).as_posix()})
    write(output / 'paired_all_images.json', paired)
    write(output / 'case_index.json', images)
    return paired, images


def generate_report(output, models, directories, masks, data, labels, inventory, pipeline, ranking,
                    raw_scans, scans, matched, frozen, guard, selection, sources):
    figures(output, scans, matched, labels)
    winner_name = selection['primary_winner']
    anchors = {n: p['baseline_recall']['summary'] for n, p in matched.items()}
    baseline, winner = matched['resnet18']['baseline_recall'], matched[winner_name]['baseline_recall']
    paired, images = case_images(output, baseline, winner | {'method': winner_name}, models, directories, masks, data)
    better, worse = sum(r['IoU_delta'] > 0 for r in paired), sum(r['IoU_delta'] < 0 for r in paired)
    base, win = baseline['summary'], winner['summary']
    reduction = 1 - win['FP'] / base['FP']
    ranking_table = {n: {'pooled': r['pooled'], 'macro_AP_37': r['macro_AP_37'],
                         'oracle_macro_IoU_37': r['oracle_macro_IoU_37'],
                         'oracle_IoU_below_05_images': r['oracle_IoU_below_05_images']} for n, r in ranking.items()}
    comparisons = {}
    for name, row in anchors.items():
        comparisons[name] = {'FP_delta_vs_baseline': row['FP'] - base['FP'],
            'Pixel_IoU_delta_vs_baseline': row['Pixel_IoU'] - base['Pixel_IoU'],
            'Pixel_Precision_delta_vs_baseline': row['Pixel_Precision'] - base['Pixel_Precision'],
            'actual_recall_gap_vs_baseline': row['Pixel_Recall'] - base['Pixel_Recall']}
    curve_support = {}
    for name in models:
        if name == 'resnet18':
            continue
        pair = [(a, b) for a, b in zip(scans['resnet18'], scans[name]) if .9 <= a['target_recall'] <= .99]
        curve_support[name] = {'high_recall_grid_points': len(pair),
            'lower_FP_points': sum(b['FP'] < a['FP'] for a, b in pair),
            'higher_pooled_IoU_points': sum(b['Pixel_IoU'] > a['Pixel_IoU'] for a, b in pair),
            'maximum_actual_recall_gap': max(abs(a['Pixel_Recall'] - b['Pixel_Recall']) for a, b in pair)}
    old_a, old_b = frozen['resnet18']['summary'], frozen['wide_resnet50']['summary']
    new_a, new_b = anchors['resnet18'], anchors['wide_resnet50']
    threshold_effect = old_b['macro_IoU_37_anomalies'] < old_a['macro_IoU_37_anomalies'] and new_b['macro_IoU_37_anomalies'] > new_a['macro_IoU_37_anomalies']
    rank_regressions = sum(ranking['wide_resnet50']['per_image'][p]['AP'] < ranking['resnet18']['per_image'][p]['AP']
                           for p in models['resnet18']['rows'] if models['resnet18']['rows'][p]['label'])
    diagnosis = {'threshold_operating_point_effect_supported': threshold_effect,
                 'candidate_per_image_AP_regressions': rank_regressions,
                 'candidate_single_image_oracle_IoU_below_05': ranking['wide_resnet50']['oracle_IoU_below_05_images'],
                 'interpretation': '阈值运行点影响有证据；仍存在逐图分数排序/定位限制。不是只换阈值就能解决全部低IoU；结构的独立因果收益证据不足。'
                  if threshold_effect and ranking['wide_resnet50']['oracle_IoU_below_05_images'] else
                  '以实际阈值曲线、逐图排序和逐图最佳IoU为依据；单一主因证据不足。'}
    lines = ['# 阈值扫描与当前方案选择', '',
        '**结论范围：现有45张开发图、5种已保存配置。未训练、未推理、未修改网络结构，未访问70张保留测试图。**', '',
        '## 明确推荐', '',
        f'按“先达到原基线{base["Pixel_Recall"]:.6%}缺陷像素召回，再选汇总Pixel IoU最高”的规则，当前开发集定位优先方案是：**{labels[winner_name]}，全局像素阈值{win["threshold"]:.10f}**。',
        f'相对同召回基线：汇总Pixel IoU {base["Pixel_IoU"]:.4f}→{win["Pixel_IoU"]:.4f}，平均IoU {base["macro_IoU_37_anomalies"]:.4f}→{win["macro_IoU_37_anomalies"]:.4f}，误圈减少{reduction:.2%}。37张异常图中{better}张提高、{worse}张下降。',
        '此“最优”只指本表固定召回要求下的5个现有配置；不包含未尝试的网络/组合、部署成本硬约束或最终泛化。现有基线和阈值配置保持原样。', '',
        '## 统一召回要求后的公平对照', '', *metrics_table(anchors, labels), '',
        '汇总Pixel IoU和Pixel Recall都由全部45图的TP/FP/FN相加后计算；正常图的FP也计入。平均IoU只对37张异常图等权，不能将两种IoU混作同一指标。', '',
        '## 每个数据代表什么', '',
        '| 数据 | 计算方法 | 直观含义 |', '|---|---|---|',
        '| Pixel Recall | TP/(TP+FN) | 真实缺陷像素中覆盖了多少；越高越少漏圈。大缺陷像素权重更大，不代表每张图召回相同。 |',
        '| Pixel IoU（汇总） | TP/(TP+FP+FN) | 预测区域与真实缺陷的交并比，漏圈和误圈都会拉低；全部图先累加计数。 |',
        '| Pixel Precision | TP/(TP+FP) | 圈出的区域中有多少确实是缺陷；越高越少把正常区域圈进去。 |',
        '| FP像素 | 正常GT且分数大于阈值的像素数 | 多圈了多少原图坐标像素，含正常图和异常图中的正常区域。 |',
        '| 误圈面积占比 | FP/(TP+FP+FN+TN) | 多圈区域占全部45张图总面积的比例。与FPR分母不同。 |',
        '| 正常像素FPR | FP/(FP+TN) | 所有正常像素中有多少被误圈。逐阈值结果也给出这个指标。 |',
        '| 预测区域污染率 | FP/(TP+FP) | 圈出区域里有多少是正常区域；等于1−Precision。 |',
        '| 平均IoU | 37张异常图各自IoU的平均 | 每张图一票，降低大缺陷主导总体指标的影响。正常图不按IoU=1计入。 |',
        '| AP | 连续分数下的精确率−召回率排序指标 | 不依赖选定的单一阈值，但不直接等于边界IoU。 |',
        '| 小面积组 | GT总面积小于全图1%的3张图 | 单独检查已知小缺陷，样本很少，不能外推普遍结论。 |',
        '| 像素阈值 | raw score > threshold | 控制掩膜，不是概率；也不是整图判定阈值13.576。不同骨干的分数尺度不可直接比较。 |', '',
        f'以本次主推荐为例：每100个真实缺陷像素覆盖约{win["Pixel_Recall"]*100:.2f}个；每100个圈出的像素中约{win["Pixel_Precision"]*100:.2f}个是真缺陷；误圈区域占全部图像面积{win["FP_area_fraction"]:.2%}。', '',
        '## 原阈值结果：保留原记录', '', *metrics_table({n: p['summary'] for n, p in frozen.items()}, labels), '',
        '上表各方法沿用各自原正常校准阈值，召回不同，不能直接据IoU列判定某个网络更好。整图判断不因本次像素阈值扫描改变。', '',
        '## 小缺陷不能退步时的敏感性对照', '',
        '附加约束：总体像素召回至少达到基线，且3张已知小面积图召回均达到100%。每种方法取满足两项条件的最大全局阈值；此时总体召回可能不同。此规则也使用开发GT，不能当部署保证。', '',
        *metrics_table({n: p['summary'] for n, p in guard.items()}, labels), '',
        f'在这项额外约束下，汇总IoU最高的候选是{labels[selection["small_guard_winner"]]}。它是单独的敏感性场景，不与前一个等召回表混合排名。', '',
        '## 阈值问题还是分数定位问题', '', diagnosis['interpretation'], '',
        f'原正常阈值下，两种骨干平均IoU：{old_a["macro_IoU_37_anomalies"]:.4f}→{old_b["macro_IoU_37_anomalies"]:.4f}；等召回后：{new_a["macro_IoU_37_anomalies"]:.4f}→{new_b["macro_IoU_37_anomalies"]:.4f}。模型与分数图不变，说明工作点会影响此前的优劣判断。',
        f'新骨干平均AP：{ranking["resnet18"]["macro_AP_37"]:.4f}→{ranking["wide_resnet50"]["macro_AP_37"]:.4f}；但有{rank_regressions}/37张的AP下降。不能说所有图的排序都更好。', '',
        '| 方法 | 汇总AP | 平均AP（37图） | 逐图独立最佳阈值的平均IoU | 逐图最佳IoU仍低于0.5的图数 |',
        '|---|---:|---:|---:|---:|']
    for name, r in ranking.items():
        lines.append(f'| {labels[name]} | {r["pooled"]["AP"]:.4f} | {r["macro_AP_37"]:.4f} | {r["oracle_macro_IoU_37"]:.4f} | {r["oracle_IoU_below_05_images"]}/37 |')
    lines += ['', '“逐图最佳阈值”在每张图全部真实缺陷分数层级上穷举，得到单阈值掩膜的精确IoU上限。它使用每张图自己的GT，不能部署、不能当最终成绩。即便这一上限仍低的图，单靠调阈值也无法使现有分数图精准定位。它不能进一步区分特征、空间分辨率或平滑各自造成多少损失。', '',
              '## 新旧骨干的具体案例', '',
              '| 开发图 | AP：旧→新 | 等召回IoU：旧→新 | 新网络逐图最佳IoU上限 |', '|---|---:|---:|---:|']
    lookup = {n: {r['image_path']: r for r in matched[n]['baseline_recall']['per_image']} for n in models}
    for path in ['metal_nut/test/scratch/000.png', 'metal_nut/test/bent/014.png', 'metal_nut/test/color/000.png', 'metal_nut/test/color/001.png']:
        a, b = (ranking[n]['per_image'][path] for n in ('resnet18', 'wide_resnet50'))
        lines.append(f'| {path} | {a["AP"]:.3f}→{b["AP"]:.3f} | {lookup["resnet18"][path]["Pixel_IoU"]:.3f}→{lookup["wide_resnet50"][path]["Pixel_IoU"]:.3f} | {b["oracle"]["max_IoU"]:.3f} |')
    lines += ['', '## 后处理一致性审计', '',
              '两骨干配置、检查点中的平滑核、生成脚本哈希已核对一致：',
              '- 相同256输入、ImageNet输入归一化；不对异常分数做min-max或概率归一化。',
              '- 相同32×32得分网格、nearest上采样至256、sigma4的33×33高斯平滑（reflect/same），再bilinear映射至原700×700，align_corners=False。',
              '- GT保留原700×700二值掩膜，图/掩膜哈希和路径对应，不插值、不裁剪、不旋转。',
              '- 保存的是阈值化前的float32连续图；扫描仅做严格大于比较，不加形态学、连通域或自适应后处理。',
              '- sigma2、512和分块方案故意改变平滑/分辨率/裁剪合并，作为不同完整配置比较；不能将它们的收益归因于换骨干。',
              '- 原始平滑前patch分数未保存，无法完整重放历史后处理；当时库文件未逐个哈希，当前2.2.0实现、记录的工程代码与保存的核共同支持本次审计。',
              '- 换骨干还换了预训练权重和特征维度，不能据此隔离“网络结构本身”的因果收益。', '',
              '## 速度与资源：沿用原实测，不重新推理', '',
              '| 对照 | 同次基线ms | 候选ms | 候选/基线耗时 | 同次基线峰值MiB | 候选峰值MiB |',
              '|---|---:|---:|---:|---:|---:|']
    costs = []
    for name, reference_key in [('wide_resnet50', 'resnet18'), ('resnet18_512', '256'),
                                ('resnet18_sigma2', 'sigma4'), ('resnet18_tiled', 'whole256')]:
        source, key = sources[name]
        a, b = source['performance'][reference_key], source['performance'][key]
        costs.append({'method': name, 'source_run': directories[name].parent.name,
                      'reference_mean_ms': a['mean_ms'], 'candidate_mean_ms': b['mean_ms'],
                      'paired_time_ratio': b['mean_ms']/a['mean_ms'],
                      'reference_peak_MiB': a['max_peak_allocated_bytes']/2**20, 'candidate_peak_MiB': b['max_peak_allocated_bytes']/2**20})
        r = costs[-1]
        lines.append(f'| {labels[name]} | {r["reference_mean_ms"]:.3f} | {r["candidate_mean_ms"]:.3f} | {r["paired_time_ratio"]:.2f}× | {r["reference_peak_MiB"]:.2f} | {r["candidate_peak_MiB"]:.2f} |')
    lines += ['', '各行来自不同历史会话，只能按同次基线比较，不能用跨行的绝对毫秒数断言精确速度排名。时间只含CUDA输入就绪后的模型处理及原坐标还原，不含解码、预处理和传输；显存为PyTorch分配峰值。阈值调整不会省去骨干和检索计算。', '',
              '## 可复核范围与缺失信息', '',
              '两骨干共90张连续分数图；附加三种历史配置共135张，均为同45图、同GT、同坐标。37张缺陷GT文件存在，8张正常图按无缺陷生成全零GT，与原清单一致。关键扫描数据齐全；没有用二值预测图替代连续图。',
              '每方法扫描501个全像素经验分位数阈值并补原阈值与端点；完整逐阈值逐图数据在各方法的JSONL文件。另用0—100%步长0.1个百分点的召回目标生成公共对照曲线。公共可达到范围为0—100%，但同分值会使某些精确召回值不可达，报告使用实际召回，不线性插值伪造像素计数。',
              '等召回工作点固定90%、95%、原基线95.904%、97%、99%；下方可切换方法和工作点，查看全部45图的指标。正常图没有真实缺陷时Recall未定义；没有预测像素时Precision未定义。',
              '数据只含一个类别、一个随机种子，开发集已被多次查看。所有阈值和逐图上限利用开发GT，只用于诊断与候选选择，不等于正式部署校准，也不支持最终泛化宣称。70张保留测试图未访问，任何模型/阈值配置均未自动替换。']
    text = '\n'.join(lines) + '\n'
    (output / '分析报告.md').write_text(text, encoding='utf-8')
    public = {'status': 'completed_development_threshold_diagnostic_not_final_evaluation', 'selection': selection,
        'labels': labels, 'matched_anchor': anchors, 'matched_comparisons': comparisons,
        'original_operating_points': {n: p['summary'] for n, p in frozen.items()},
        'small_defect_guard': {n: p['summary'] for n, p in guard.items()}, 'ranking_summary': ranking_table,
        'diagnosis': diagnosis, 'costs_paired_from_historical_runs': costs, 'curve_support_90_to_99_percent': curve_support,
        'mean_IoU_change_counts_winner_vs_baseline': {'improved': better, 'worse': worse, 'equal': len(paired)-better-worse},
        'inventory': inventory, 'pipeline_audit': pipeline, 'automatic_adoption': False,
        'held_out_images_opened': 0, 'new_training': False, 'new_inference': False}
    write(output / 'public_summary.json', public)
    rows_data = {n: {key: p['per_image'] for key, p in values.items()} |
                 {'original': frozen[n]['per_image'], 'small_guard': guard[n]['per_image']} for n, values in matched.items()}
    script_data = json.dumps({'labels': labels, 'rows': rows_data}, ensure_ascii=False).replace('<', '\\u003c')
    ui = '''<h2>全部45张图的逐图结果</h2><p>切换方法与工作点。IoU、Recall、Precision和面积均为该图原坐标的像素统计；正常图不纳入37图平均IoU。</p><select id="method"></select> <select id="point"><option value="baseline_recall">基线95.904%总体召回工作点</option><option value="recall90">90%总体召回</option><option value="recall95">95%总体召回</option><option value="recall97">97%总体召回</option><option value="recall99">99%总体召回</option><option value="small_guard">小缺陷100%保护</option><option value="original">原正常校准阈值</option></select><div style="overflow:auto"><table><thead><tr><th>图片</th><th>Recall</th><th>IoU</th><th>Precision</th><th>FP</th><th>FN</th><th>误圈占图面积</th></tr></thead><tbody id="rows"></tbody></table></div>'''
    js = '''<script>const D=DATA;const M=document.getElementById('method'),P=document.getElementById('point');Object.entries(D.labels).forEach(([k,v])=>{const o=document.createElement('option');o.value=k;o.textContent=v;M.append(o)});function pct(v){return v===null?'未定义':(v*100).toFixed(3)+'%'}function draw(){const t=document.getElementById('rows');t.replaceChildren();D.rows[M.value][P.value].forEach(r=>{const tr=document.createElement('tr');[r.image_path,pct(r.Pixel_Recall),r.Pixel_IoU===null?'未定义':r.Pixel_IoU.toFixed(4),pct(r.Pixel_Precision),r.FP,r.FN,pct(r.FP_area_fraction)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});t.append(tr)})}M.onchange=draw;P.onchange=draw;draw()</script>'''.replace('DATA', script_data)
    cards = ''.join('<details><summary>' + html.escape(r['image_path']) + '</summary><img loading="lazy" src="' + html.escape(r['path']) + '"></details>' for r in images)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>阈值扫描与方案选择</title><style>body{max-width:1450px;margin:30px auto;padding:20px;font:16px/1.7 "Microsoft YaHei",sans-serif;color:#233749;background:#fafcff}h1,h2{color:#113954}table{border-collapse:collapse;font-size:14px;width:100%;background:white}th,td{border:1px solid #dce3e9;padding:9px;text-align:left}th{background:#edf3f8}img{width:100%}select{padding:10px;margin:8px}summary{padding:12px;cursor:pointer}a{color:#176ca6}</style><h1>阈值扫描与当前最优候选</h1><p><a href="分析报告.md">报告原文</a> · <a href="public_summary.json">摘要数据</a> · <a href="pipeline_audit.json">后处理审计</a></p>'
    page += html_summary(text) + '<h2>IoU—Recall：汇总口径一致</h2><img src="IoU-Recall.png"><h2>FP—Recall</h2><img src="FP-Recall.png"><img src="Precision与误圈占比.png">' + ui + '<h2>主推荐相对基线：收益与退步案例</h2><p>按逐图IoU变化取前三改善、前三退步，另含全部小面积案例；不是只挑成功图。绿=正确覆盖、红=误圈、蓝=漏圈。</p>' + cards + js + '</html>'
    (output / '对照报告.html').write_text(page, encoding='utf-8')
