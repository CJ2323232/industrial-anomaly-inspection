"""One frozen held-out evaluation, with a fixed historical R18 reference.
Never fits, calibrates, scans thresholds, or changes any previous experiment.
"""
import argparse
import gc
import html
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from inspection_runtime import ROOT, DATA
from baseline_utils import file_sha256, load_manifest, safe_file, binary_counts, summarize_counts, binary_metrics
from threshold_scan_metrics import rank_metrics
from resolution_metrics import evaluate_records
from analyze_matched_recall import now, read, write, html_summary

FREEZE = ROOT/'02代码与配置/configs/decisions/metal_nut-wrn50-frozen-v1.json'
BASE_POLICY = ROOT/'02代码与配置/configs/decisions/metal_nut-development-reference-v1.json'
BASE_MODEL = ROOT/'04实验结果/01模型与检索实验/baseline_20261016_204723_088238/model.pt'
MANIFEST = ROOT/'02代码与配置/configs/splits/metal_nut-v1.json'
DEST = ROOT/'04实验结果/15冻结版本独立测试'
KEYS=('TP','FP','FN','TN')

def verify_freeze():
    freeze=read(FREEZE)
    for relative,digest in freeze['file_sha256'].items():
        if file_sha256(safe_file(ROOT,relative)) != digest:
            raise ValueError('Frozen identity mismatch: '+relative)
    base=read(BASE_POLICY)
    assert file_sha256(BASE_MODEL)==base['checkpoint_sha256']
    manifest=load_manifest(MANIFEST)
    assert len(manifest['splits']['held_out'])==70
    assert set(manifest['splits']['held_out']).isdisjoint(manifest['splits']['development'])
    return freeze,base,manifest

def mask_for(record):
    if record['mask_path']:
        file=safe_file(DATA,record['mask_path'])
        if file_sha256(file)!=record['mask_sha256']:raise ValueError('GT identity mismatch')
        with Image.open(file) as im: mask=np.asarray(im)>0
    else: mask=np.zeros((record['height'],record['width']),bool)
    return mask

def wilson(success,n):
    if not n:return None
    z=1.959963984540054;p=success/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-half),float(mid+half)]

def run_model(name,cfg,manifest,output):
    import torch
    from run_trusted_baseline import load_checkpoint,make_transform,load_tensor,predict
    folder=output/name;folder.mkdir(exist_ok=True)
    journal=folder/'predictions.jsonl'
    rows=[json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()] if journal.exists() else []
    existing={r['image_path']:r for r in rows}
    paths=manifest['splits']['held_out']
    assert len(rows)==len(existing) and set(existing)<=set(paths)
    model=None
    if len(existing)<len(paths):
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model,saved=load_checkpoint(safe_file(ROOT,cfg['checkpoint']),device)
        model.requires_grad_(False)
        assert saved['manifest_sha256']==manifest['manifest_sha256']
        transform=make_transform(saved['model_config'])
        if name=='wide_resnet50':
            assert saved['model_config']==read(FREEZE)['preprocessing']
    for index,path in enumerate(paths):
        if path in existing:continue
        record=manifest['files'][path]
        file=safe_file(DATA,path)
        assert file_sha256(file)==record['image_sha256']
        mask=mask_for(record)
        start=time.perf_counter()
        tensor=load_tensor(DATA,path,transform).to(device)
        score,array=predict(model,tensor,(record['height'],record['width']))
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed=(time.perf_counter()-start)*1000
        counts=binary_counts(mask.ravel(),array.ravel(),cfg['pixel_threshold'])
        ranks=rank_metrics(np.sort(array[mask]),np.sort(array[~mask]))
        dest=folder/'maps'/Path(path).with_suffix('.npz')
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,anomaly_map=array)
        row={'image_path':path,'split':'held_out','split_label':'保留独立测试集',
             'label':record['label'],'defect_type':record['defect_type'],'image_sha256':record['image_sha256'],
             'mask_sha256':record['mask_sha256'],'area_ratio':float(mask.mean()),'raw_score':score,
             'pred_label':int(score>cfg['image_threshold']),'pixel_counts':counts,
             'pixel_AP':ranks['AP'],'pixel_AUROC':ranks['AUROC'],
             'map_path':dest.relative_to(output).as_posix(),'processing_ms':elapsed,
             'device':str(device)}
        with journal.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        rows.append(row);existing[path]=row
        if (index+1)%10==0:print(name,index+1,'/70',flush=True)
    if model is not None:del model
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    rows=[existing[p] for p in paths]
    # Exact rank metrics from saved continuous scores; no threshold search.
    pos=[];neg=[]
    for r in rows:
        mask=mask_for(manifest['files'][r['image_path']])
        with np.load(safe_file(output,r['map_path']),allow_pickle=False) as z:array=z['anomaly_map']
        assert array.dtype==np.float32 and array.shape==mask.shape and np.isfinite(array).all()
        assert binary_counts(mask.ravel(),array.ravel(),cfg['pixel_threshold'])==r['pixel_counts']
        pos.append(array[mask]);neg.append(array[~mask])
    positive=np.sort(np.concatenate(pos));negative=np.sort(np.concatenate(neg))
    summary=evaluate_records(rows)
    summary['pooled_pixel_ranking']=rank_metrics(positive,negative)
    summary['image_continuous_metrics']=binary_metrics([r['label'] for r in rows],[r['raw_score'] for r in rows],cfg['image_threshold'])
    counts=summary['pixel_counts']
    summary['FP_area_fraction']=counts['FP']/sum(counts.values())
    ims=summary['image_counts']
    summary['image_FPR_wilson95']=wilson(ims['FP'],ims['FP']+ims['TN'])
    summary['image_FNR_wilson95']=wilson(ims['FN'],ims['FN']+ims['TP'])
    small=[r for r in rows if r['label'] and r['area_ratio']<.01]
    summary['small_area_under_1percent']={'n':len(small),'pixel_metrics':summarize_counts({k:sum(r['pixel_counts'][k] for r in small) for k in KEYS}),
        'macro_IoU':float(np.mean([summarize_counts(r['pixel_counts'])['IoU'] for r in small])) if small else None}
    summary['image_errors']=[{'image_path':r['image_path'],'label':r['label'],'prediction':r['pred_label'],'raw_score':r['raw_score']} for r in rows if r['label']!=r['pred_label']]
    write(folder/'predictions.json',rows);write(folder/'summary.json',summary)
    print(name,'metrics completed',flush=True)
    return rows,summary

def figures_and_report(output,models,rows,summaries,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    indexed={n:{r['image_path']:r for r in rs} for n,rs in rows.items()}
    per_image=[];cards=[]
    for index,path in enumerate(manifest['splits']['held_out']):
        record=manifest['files'][path];mask=mask_for(record)
        with Image.open(safe_file(DATA,path)) as im:original=np.asarray(im)
        fig,axes=plt.subplots(1,4,figsize=(14,4.5))
        axes[0].imshow(original);axes[0].set_title('Original')
        axes[1].imshow(mask,cmap='gray',vmin=0,vmax=1);axes[1].set_title('Ground truth')
        pair={'image_path':path,'split':'held_out','label':record['label'],'defect_type':record['defect_type']}
        for ax,name,title in [(axes[2],'resnet18','R18 baseline'),(axes[3],'wide_resnet50','WRN50 frozen')]:
            row=indexed[name][path]
            with np.load(safe_file(output,row['map_path']),allow_pickle=False) as z:score_map=z['anomaly_map']
            predicted=score_map>models[name]['pixel_threshold']
            error=np.zeros((*mask.shape,3),np.uint8);error[predicted&mask]=(0,185,95);error[predicted&~mask]=(240,70,65);error[~predicted&mask]=(50,125,255)
            ax.imshow(error);m=summarize_counts(row['pixel_counts'])
            iou='n/a' if not record['label'] else f'{m["IoU"]:.3f}'
            ax.set_title(f'{title}: IoU {iou}\nFP {m["FP"]}, FN {m["FN"]}')
            pair[name]={'image_prediction':row['pred_label'],'raw_score':row['raw_score'],**m}
        pair['IoU_delta']=pair['wide_resnet50']['IoU']-pair['resnet18']['IoU'] if record['label'] else None
        dest=output/'可视化_独立测试集'/Path(path)
        dest.parent.mkdir(parents=True,exist_ok=True)
        for ax in axes:ax.axis('off')
        fig.suptitle('HELD-OUT TEST | '+path+'\nGreen=TP / red=FP / blue=FN | frozen thresholds; no tuning')
        fig.tight_layout(rect=(0,0,1,.84));fig.savefig(dest,dpi=110);plt.close(fig)
        pair['figure']=dest.relative_to(output).as_posix();per_image.append(pair)
        cards.append(f'<details><summary>【独立测试集】{html.escape(path)} | {"正常" if not record["label"] else record["defect_type"]}</summary><img loading="lazy" src="{html.escape(pair["figure"])}"></details>')
    write(output/'逐图对照.json',per_image)
    b,w=summaries['resnet18'],summaries['wide_resnet50']
    anomaly=[x for x in per_image if x['label']]
    changes={'improved':sum(x['IoU_delta']>0 for x in anomaly),'worse':sum(x['IoU_delta']<0 for x in anomaly),'equal':sum(x['IoU_delta']==0 for x in anomaly)}
    write(output/'配对变化.json',changes)
    def f(x):return '未定义' if x is None else f'{x:.4f}'
    lines=['# 冻结版本：70张独立测试结果','',
      '本次使用首次开放的保留测试子集，模型和阈值在读取测试图片之前已固定。WRN50为用户已选方案，R18仅作预先指定的历史参照；未扫描阈值、重训或据测试结果选模型。','',
      f'样本构成：{w["normal_images"]}张正常、{w["anomalous_images"]}张异常，共70张。两种方法使用完全相同图片、原尺寸GT和像素统计。','',
      '|指标|R18历史基线|WRN50冻结版本|','|---|---:|---:|']
    vals=[('整图误报 / 正常图',f'{b["image_counts"]["FP"]}/{b["normal_images"]}',f'{w["image_counts"]["FP"]}/{w["normal_images"]}'),
      ('整图漏检 / 异常图',f'{b["image_counts"]["FN"]}/{b["anomalous_images"]}',f'{w["image_counts"]["FN"]}/{w["anomalous_images"]}'),
      ('整图AUROC',f(b['image_continuous_metrics']['AUROC']),f(w['image_continuous_metrics']['AUROC'])),
      ('整图AP',f(b['image_continuous_metrics']['AP']),f(w['image_continuous_metrics']['AP'])),
      ('Pixel Recall',f(b['pixel_metrics']['recall']),f(w['pixel_metrics']['recall'])),
      ('汇总Pixel IoU',f(b['pixel_metrics']['IoU']),f(w['pixel_metrics']['IoU'])),
      ('Pixel Precision',f(b['pixel_metrics']['precision']),f(w['pixel_metrics']['precision'])),
      ('FP误圈像素',str(b['pixel_counts']['FP']),str(w['pixel_counts']['FP'])),
      ('FP / 全部像素',f(b['FP_area_fraction']),f(w['FP_area_fraction'])),
      ('异常图等权平均IoU',f(b['anomaly_macro']['pixel_IoU']),f(w['anomaly_macro']['pixel_IoU'])),
      ('异常图等权平均AP',f(b['anomaly_macro']['pixel_AP']),f(w['anomaly_macro']['pixel_AP'])),
      ('汇总Pixel AP',f(b['pooled_pixel_ranking']['AP']),f(w['pooled_pixel_ranking']['AP'])),
      ('汇总Pixel AUROC',f(b['pooled_pixel_ranking']['AUROC']),f(w['pooled_pixel_ranking']['AUROC'])),
      ('小面积(<1%)图片数量',str(b['small_area_under_1percent']['n']),str(w['small_area_under_1percent']['n'])),
      ('小面积像素Recall',f(b['small_area_under_1percent']['pixel_metrics']['recall']),f(w['small_area_under_1percent']['pixel_metrics']['recall'])),
      ('小面积平均IoU',f(b['small_area_under_1percent']['macro_IoU']),f(w['small_area_under_1percent']['macro_IoU']))]
    lines += [f'|{k}|{a}|{c}|' for k,a,c in vals]
    lines += ['', '## 直接结论','',
      f'WRN相对R18：汇总IoU变化{w["pixel_metrics"]["IoU"]-b["pixel_metrics"]["IoU"]:+.4f}，Pixel Recall变化{100*(w["pixel_metrics"]["recall"]-b["pixel_metrics"]["recall"]):+.3f}个百分点，FP变化{w["pixel_counts"]["FP"]-b["pixel_counts"]["FP"]:+,}像素。',
      f'异常图逐图IoU：{changes["improved"]}张提高、{changes["worse"]}张下降、{changes["equal"]}张相同。两者实际召回不同，这不是等召回对照。',
      '本次是冻结工作点的独立测试，不会因为某项结果不好就换阈值重新报告。完整70张结果和错误案例全部保留。','',
      '## WRN整图错误清单','']
    lines += [f'- {r["image_path"]}：真实{r["label"]}，预测{r["prediction"]}，原始分数{r["raw_score"]:.6f}。' for r in w['image_errors']] or ['没有整图分类错误。']
    lines += ['', '## 指标怎么理解','',
      '- Pixel Recall：真实缺陷像素中圈到了多少。Precision：圈出的像素中有多少是真的缺陷。',
      '- IoU：TP/(TP+FP+FN)。汇总IoU先累计全部70张的计数，平均IoU对异常图等权；两者不能混用。',
      '- FP是误圈的正常像素。误圈面积占比的分母是全部70张图像素，不是预测面积。',
      '- AP/AUROC使用连续分数评价排序，不代表某个固定阈值能实现的准确率。',
      '- 0误报或0漏检也不能推断总体错误率为0；JSON中附整图错误率Wilson 95%区间。','',
      '## 适用范围与下一步','',
      '本次能评价该MVTec AD metal_nut固定划分上的未参与调参样本表现；不能证明新工厂、相机、照明、零件或所有小缺陷都可靠。阈值使用了开发集缺陷GT，不能宣传成完全不使用缺陷标注选参数。',
      '原来的开发图仍是开发图，不能与这70张合并后当成新的独立测试分数。后续若按这些测试错误继续改算法，需要额外未见数据验证新版本。',
      '模型耗时和显存沿用既有实验参照，当前脚本的单次处理时间不作为新的稳态性能基准。','',
      '图片来源说明见项目文档“图片属于哪个数据集.html”。本页所有下方案例均明确标注为独立测试集。']
    md='\n'.join(lines)+'\n';(output/'测试结果说明.md').write_text(md,encoding='utf-8')
    css='<style>body{max-width:1300px;margin:30px auto;padding:0 20px;font:16px/1.75 system-ui;color:#193449;background:#f5f8fb}table{width:100%;border-collapse:collapse;background:white}td,th{border:1px solid #d7e0e8;padding:8px;text-align:left}th{background:#deedf6}img{width:100%}details{background:white;margin:12px 0;padding:12px}h1,h2{color:#126080}</style>'
    (output/'独立测试报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>冻结版本独立测试</title>'+css+'<h1>冻结版本：70张独立测试结果</h1>'+html_summary(md)+'<h2>全部70张测试图片</h2>'+''.join(cards)+'</html>',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Explicit authorization to evaluate held-out images')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    freeze,base,manifest=verify_freeze()
    if not args.run:print('PREFLIGHT_OK; held-out images not opened. Pass --run only when authorized.');return
    models={'resnet18':{'checkpoint':BASE_MODEL.relative_to(ROOT).as_posix(),'image_threshold':base['image_threshold'],'pixel_threshold':base['pixel_threshold']},
      'wide_resnet50':{'checkpoint':freeze['checkpoint'],'image_threshold':freeze['image_threshold'],'pixel_threshold':freeze['pixel_threshold']}}
    DEST.mkdir(parents=True,exist_ok=True)
    marker=DEST/'metal_nut-wrn50-frozen-v1_first_evaluation.json'
    if args.resume:
        output=args.resume.resolve()
        assert output.is_relative_to(DEST) and output.is_dir()
        protocol=read(output/'protocol.json')
        assert protocol['frozen_sha256']==file_sha256(FREEZE) and protocol['models']==models
        assert read(marker)['output']==output.relative_to(ROOT).as_posix()
        if (output/'status.json').exists() and read(output/'status.json')['status']=='completed':
            print('Already completed; no repeat evaluation.');return
    else:
        output=DEST/('heldout_'+now().strftime('%Y%m%d_%H%M%S_%f'))
        with marker.open('x',encoding='utf-8') as f:json.dump({'output':output.relative_to(ROOT).as_posix(),'created_at':now().isoformat()},f,ensure_ascii=False,indent=2)
        output.mkdir()
        protocol={'frozen_sha256':file_sha256(FREEZE),'frozen_version':freeze['version'],'models':models,
          'manifest_sha256':manifest['manifest_sha256'],'test_images':manifest['splits']['held_out'],
          'model_identities':{n:file_sha256(safe_file(ROOT,c['checkpoint'])) for n,c in models.items()},
          'baseline_policy_sha256':file_sha256(BASE_POLICY),'selection_before_test':True,
          'authorized_by_user':True,'threshold_scan':False,'training':False,'read_before_protocol':False,
          'metrics':['fixed image confusion','fixed pooled pixel confusion','macro localization','exact AP/AUROC','small-area groups','per-defect','all individual images'],
          'inference_mode':'FP32 reference retrieval, TF32 disabled, batch1, raw maps, no morphological postprocessing',
          'source_sha256':file_sha256(Path(__file__)),'started_at':now().isoformat()}
        write(output/'protocol.json',protocol)
    write(output/'status.json',{'status':'running','updated_at':now().isoformat()})
    import torch
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    rows={};summaries={}
    for name,config in models.items():rows[name],summaries[name]=run_model(name,config,manifest,output)
    verify_freeze()
    write(output/'summary.json',summaries)
    figures_and_report(output,models,rows,summaries,manifest)
    write(output/'status.json',{'status':'completed','completed_at':now().isoformat(),'distinct_held_out_images_evaluated':70,'model_forward_passes':140,'frozen_identities_unchanged':True,'thresholds_unchanged':True,'training_runs':0})
    print('COMPLETED',str(output),flush=True)
    print(json.dumps({n:{k:s[k] for k in ['image_counts','pixel_metrics','anomaly_macro','small_area_under_1percent']} for n,s in summaries.items()},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
