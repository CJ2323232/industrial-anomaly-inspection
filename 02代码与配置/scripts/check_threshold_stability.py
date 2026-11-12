"""Image-level five-fold development threshold stability; cached maps only.

This checks threshold estimation, not training or independent generalization:
the 45 development images have already informed earlier project choices.
"""
import argparse
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
if __package__:
    from .analyze_matched_recall import ROOT, SOURCE, DATA, read, write, now, html_summary
    from .baseline_utils import load_manifest, safe_file, file_sha256
    from .threshold_scan_metrics import pixel_metrics, cutoff_at_count
else:
    from analyze_matched_recall import ROOT, SOURCE, DATA, read, write, now, html_summary
    from baseline_utils import load_manifest, safe_file, file_sha256
    from threshold_scan_metrics import pixel_metrics, cutoff_at_count


def count_at(pos, neg, threshold):
    tp = len(pos) - int(np.searchsorted(pos, threshold, side='right'))
    fp = len(neg) - int(np.searchsorted(neg, threshold, side='right'))
    return {'TP': tp, 'FP': fp, 'FN': len(pos)-tp, 'TN': len(neg)-fp}


def aggregate(rows):
    m = pixel_metrics({k: sum(r[k] for r in rows) for k in ('TP','FP','FN','TN')})
    anomalous = [r for r in rows if r['TP'] + r['FN'] > 0]
    small = [r for r in anomalous if r['area_ratio'] < .01]
    m['macro_IoU'] = float(np.mean([r['Pixel_IoU'] for r in anomalous]))
    m['small_n'] = len(small)
    m['small_macro_IoU'] = float(np.mean([r['Pixel_IoU'] for r in small])) if small else None
    m['small_Recall'] = sum(r['TP'] for r in small)/sum(r['TP']+r['FN'] for r in small) if small else None
    m['normal_FP'] = sum(r['FP'] for r in rows if r['TP']+r['FN'] == 0)
    return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    args=parser.parse_args()
    source=args.analysis.resolve()
    if not source.is_relative_to(ROOT): raise ValueError('Canonical project only')
    inventory=read(source/'data_inventory.json')
    reference=read(source/'matched_working_points.json')
    manifest=load_manifest(SOURCE/'split_manifest.json')
    paths=manifest['splits']['development']
    base={r['image_path']:r for r in reference['resnet18']['baseline_recall']['per_image']}
    assert len(paths)==45 and set(paths)==set(base)
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=20261026)
    folds=[(tr.tolist(),va.tolist()) for tr,va in splitter.split(paths,[base[p]['defect_type'] for p in paths])]
    assert sorted(i for _,va in folds for i in va)==list(range(45))
    output=ROOT/'04实验结果/13阈值稳定性复核'/('threshold_cv_'+now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    targets=['0.95','0.97','0.99']
    write(output/'protocol.json',{'scope':'development-only threshold-estimation cross-validation; not independent model validation',
        'seed':20261026,'targets':targets,'folds':[{'selection':[paths[i] for i in tr],'validation':[paths[i] for i in va]} for tr,va in folds],
        'selection_rule':'largest global cutoff attaining target pooled pixel Recall on selection images; float32 strict >',
        'models_frozen':True,'held_out_images_opened':0,'training_runs':0,'uses_development_defect_GT':True,
        'source':source.relative_to(ROOT).as_posix(), 'manifest_sha256':manifest['manifest_sha256']})
    masks={}
    for p in paths:
        rec=manifest['files'][p]
        if rec['mask_path']:
            f=safe_file(DATA,rec['mask_path']); assert file_sha256(f)==rec['mask_sha256']
            with Image.open(f) as im: masks[p]=np.asarray(im)>0
        else: masks[p]=np.zeros((rec['height'],rec['width']),bool)
    result={}; verified=0
    for name,inv in inventory.items():
        scores={}; hashes=[]
        for p in paths:
            f=safe_file(ROOT/inv['source'],'maps/development/'+Path(p).with_suffix('.npz').as_posix())
            with np.load(f,allow_pickle=False) as z: a=z['anomaly_map']
            assert a.dtype==np.float32 and a.shape==masks[p].shape and np.isfinite(a).all()
            scores[p]=(np.sort(a[masks[p]]),np.sort(a[~masks[p]]))
            hashes.append({'path':f.relative_to(ROOT).as_posix(),'sha256':file_sha256(f)})
        result[name]={}
        for target in targets:
            per_image=[]; fold_summaries=[]
            for fold,(tr,va) in enumerate(folds):
                assert not set(tr)&set(va)
                positive=np.sort(np.concatenate([scores[paths[i]][0] for i in tr]))
                required=int((Decimal(target)*len(positive)).to_integral_value(rounding=ROUND_CEILING))
                threshold=cutoff_at_count(positive,np.empty(0,np.float32),required)
                selection_counts=[count_at(*scores[paths[i]],threshold) for i in tr]
                selection=pixel_metrics({k:sum(r[k] for r in selection_counts) for k in ('TP','FP','FN','TN')})
                assert selection['TP']>=required
                part=[]
                for i in va:
                    p=paths[i]; pos,neg=scores[p]
                    m=pixel_metrics(count_at(pos,neg,threshold))
                    # Independent boolean-array count check, preserving strict > ties.
                    assert m['TP']==int(np.count_nonzero(pos>threshold)) and m['FP']==int(np.count_nonzero(neg>threshold))
                    verified+=1
                    r=m|{'image_path':p,'fold':fold,'threshold':threshold,'area_ratio':base[p]['area_ratio'],'defect_type':base[p]['defect_type']}
                    part.append(r)
                per_image.extend(part)
                fold_summaries.append({'fold':fold,'threshold':threshold,'selection':selection,'validation':aggregate(part)})
            assert len(per_image)==45 and {r['image_path'] for r in per_image}==set(paths)
            result[name][target]={'summary':aggregate(per_image),'folds':fold_summaries,'per_image':per_image,
                'threshold_range':[min(r['threshold'] for r in fold_summaries),max(r['threshold'] for r in fold_summaries)],
                'validation_recall_range':[min(r['validation']['Pixel_Recall'] for r in fold_summaries),max(r['validation']['Pixel_Recall'] for r in fold_summaries)]}
        write(output/(name+'_input_hashes.json'),hashes)
        print('completed',name,flush=True)
    write(output/'results.json',result)
    lines=['# 阈值稳定性：用4份选阈值，留1份验证','',
        '已固定模型，用已有45张开发集做按图片、缺陷类型分层的5折复核。每个阈值只看其余4折的GT，验证折不参与该阈值选择。95%、97%、99%是选择折目标，验证折实际召回可能达不到。','',
        '该开发集已被多次查看，因此这只是阈值稳定性检查，不是独立泛化验证。没有读取70张保留测试图，没有训练或运行新推理。','',
        '|选择折目标Recall|方案|验证Pixel Recall|验证汇总IoU|验证Precision|FP像素|误圈占全图|37图平均IoU|小面积Recall|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for target in targets:
        for name in inventory:
            s=result[name][target]['summary']
            lines.append(f'|{float(target):.0%}|{inventory[name]["label"]}|{s["Pixel_Recall"]:.3%}|{s["Pixel_IoU"]:.4f}|{s["Pixel_Precision"]:.3%}|{s["FP"]:,}|{s["FP_area_fraction"]:.3%}|{s["macro_IoU"]:.4f}|{s["small_Recall"]:.3%}|')
    comparisons={}
    for target in targets:
        a=result['resnet18'][target]['summary']; b=result['wide_resnet50'][target]['summary']
        comparisons[target]={'WRN_minus_R18_IoU':b['Pixel_IoU']-a['Pixel_IoU'],
            'WRN_minus_R18_Recall':b['Pixel_Recall']-a['Pixel_Recall'],
            'WRN_FP_change_fraction':b['FP']/a['FP']-1}
    write(output/'comparison.json',comparisons)
    lines+=['','## 实际结果如何解读','']
    for t,c in comparisons.items():
        lines.append(f'- 选择折目标{float(t):.0%}：WRN相对R18，验证IoU变化{c["WRN_minus_R18_IoU"]:+.4f}，实际Recall变化{c["WRN_minus_R18_Recall"]*100:+.3f}个百分点，FP变化{c["WRN_FP_change_fraction"]:+.2%}。')
    lines+=['','这些比较使用相同划分和相同选择目标，但实际验证召回不完全相等，不能写成严格等召回优势。严格等召回结论仍请看前一次阈值扫描。','',
        '## 阈值跨折波动','', '|方案|选择目标|5个阈值范围|验证折Recall最低—最高|','|---|---:|---:|---:|']
    for n in inventory:
        for t in targets:
            r=result[n][t]; low,high=r['threshold_range']; rl,rh=r['validation_recall_range']
            lines.append(f'|{inventory[n]["label"]}|{float(t):.0%}|{low:.6f}—{high:.6f}|{rl:.2%}—{rh:.2%}|')
    lines+=['','## 使用方式','',
        '保持成本基线和定位候选两个角色，不自动换模型。先检查候选在三个预设目标下的Recall、FP、平均IoU和小缺陷是否同时可接受；不能只选最有利的一行。',
        'Pixel Recall=TP/(TP+FN)，IoU=TP/(TP+FP+FN)，Precision=TP/(TP+FP)，误圈占全图=FP/(45×700×700)。平均IoU对37张异常图等权；小面积Recall合计3张小面积图。',
        '每张图仅产生一份折外预测，先累计45张图TP/FP/FN/TN再计算主指标；不平均5个折指标。完整逐图数值、划分、阈值和文件哈希均保存在同目录JSON中。']
    text='\n'.join(lines)+'\n'; (output/'稳定性报告.md').write_text(text,encoding='utf-8')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>阈值稳定性复核</title><style>body{max-width:1350px;margin:35px auto;padding:0 20px;font:16px/1.7 system-ui;color:#183449;background:#f6f8fa}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #d8e2eb}th{background:#dfebf5}h1,h2{color:#126080}</style><h1>阈值稳定性复核</h1>'+html_summary(text)+'</html>'
    (output/'稳定性报告.html').write_text(page,encoding='utf-8')
    write(output/'verification.json',{'status':'completed','verified_counts':verified,'fold_validation_union':45,
        'validation_overlap':0,'held_out_images_opened':0,'source_inventory_sha256':file_sha256(source/'data_inventory.json')})
    print(str(output),flush=True)
    print(comparisons,flush=True)


if __name__=='__main__': main()
