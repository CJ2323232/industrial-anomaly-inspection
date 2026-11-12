"""One fixed WRN50_v2 evaluation on the previously used 70-image subset."""
import argparse,html,json
from pathlib import Path
import numpy as np
import evaluate_frozen_holdout as original
from baseline_utils import file_sha256,load_manifest,safe_file
from analyze_matched_recall import now,read,write
ROOT=Path(__file__).resolve().parents[2]
FREEZE=ROOT/'02代码与配置/configs/decisions/WRN50_v2.json'
DEST=ROOT/'04实验结果/17WRN50_v2冻结复测'
PRIOR=ROOT/'04实验结果/15冻结版本独立测试/heldout_20261026_130153_088296'
MARKER=DEST/'WRN50_v2_once.json'
def verify():
    f=read(FREEZE)
    for p,h in f['file_sha256'].items():
        if file_sha256(safe_file(ROOT,p))!=h:raise ValueError('Frozen identity mismatch: '+p)
    m=load_manifest(ROOT/'02代码与配置/configs/splits/metal_nut-v1.json')
    assert m['manifest_sha256']==f['manifest_sha256'] and len(m['splits']['held_out'])==70
    return f,m
def report(out,f,rows,summary,oldrows,olds,checks):
    ss=[olds['resnet18'],olds['wide_resnet50'],summary]
    table=['|指标|R18历史|WRN50/v1历史|WRN50_v2本次|','|---|---:|---:|---:|']
    fields=[
      ('正常图误报 /14',[str(s['image_counts']['FP']) for s in ss]),
      ('异常图漏检 /56',[str(s['image_counts']['FN']) for s in ss]),
      ('图像Recall',[f"{s['image_continuous_metrics']['recall']:.4%}" for s in ss]),
      ('正常图FPR',[f"{s['image_continuous_metrics']['false_positive_rate']:.4%}" for s in ss]),
      ('图像Precision',[f"{s['image_continuous_metrics']['precision']:.4%}" for s in ss]),
      ('Pixel Recall',[f"{s['pixel_metrics']['recall']:.4%}" for s in ss]),
      ('汇总Pixel IoU',[f"{s['pixel_metrics']['IoU']:.6f}" for s in ss]),
      ('Pixel Precision',[f"{s['pixel_metrics']['precision']:.4%}" for s in ss]),
      ('FP正常像素',[f"{s['pixel_counts']['FP']:,}" for s in ss]),
      ('异常图等权平均IoU',[f"{s['anomaly_macro']['pixel_IoU']:.6f}" for s in ss]),
      ('图像AUROC',[f"{s['image_continuous_metrics']['AUROC']:.6f}" for s in ss])]
    table += ['|'+k+'|'+'|'.join(v)+'|' for k,v in fields]
    changes=[dict(image_path=r['image_path'],label=r['label'],v1_prediction=oldrows[r['image_path']]['pred_label'],v2_prediction=r['pred_label'],raw_score=r['raw_score']) for r in rows if r['pred_label']!=oldrows[r['image_path']]['pred_label']]
    write(out/'整图判定变化.json',changes)
    lines=['# WRN50_v2 冻结配置70图固定工作点评估','','## 结果性质','',
      '**本次是此前已使用测试子集上的固定版本复测，不是v2全新独立测试。** 70图曾用于v1评估，v1误报诊断促成了v2阈值工作。v2阈值由45图开发集选择；本次运行前冻结模型及两阈值，运行中未扫描阈值、未训练，结束后不继续调参。','',
      f"模型：{f['backbone']}；图像阈值 **{f['image_threshold']:.16f}**；像素阈值 **{f['pixel_threshold']:.16f}**。14正常、56异常。R18/v1引用历史结果，仅WRN50_v2本次运行70次前向。",'','## 指标对照','',*table,'',
      '图像Recall=检出的异常图/56；正常图FPR=误报正常图/14；图像Precision=报警图中真实异常图的比例。Pixel Recall按缺陷像素计数；IoU=TP/(TP+FP+FN)，先汇总70图计数；FP为误圈正常像素数。异常图平均IoU对56图等权。','',
      '## 阈值来源与审计','',
      '图像阈值是开发集图像标签选择的阈值，不是仅正常校准所得。规则为开发集FN=0→FP最少→最佳分类平台中点；由最高正常分数2.2242748737335205与最低异常分数2.23604416847229的中点得到。开发集0/8正常误报、0/37异常漏检已复算。','',
      '**计算遵循当次扫描前写入protocol的规则，但不能称作查看开发数据之前的严格预注册。** 本会话在制定具体规则前，已经查看开发正常/异常分数边界。更早的v1测试结果也已揭示，必须保留这些开发历史。像素阈值沿用开发集GT选择的2.103178024291992。','',
      '先前20次五折开发诊断中，这条选择规则每轮平均误报1/8、漏检1.05/37，未显示稳健的零错误。本次没有依据测试结果重跑诊断或改变规则。','',
      '## 本次整图判定变化','']
    lines += [f"- {r['image_path']}：真实{r['label']}，v1预测{r['v1_prediction']}→v2预测{r['v2_prediction']}，分数{r['raw_score']:.12f}。" for r in changes] or ['无整图判定变化。']
    lines += ['','## 本次全部错误','']
    lines += [f"- {r['image_path']}：真实{r['label']}，预测{r['prediction']}，分数{r['raw_score']:.12f}。" for r in summary['image_errors']] or ['本次固定70图未出现整图分类错误；这不表示真实部署错误率为0。']
    lines += ['','## 核验与证据边界','',
      f"与WRN50/v1配对：70图原始图像分数最大绝对差={checks['max_abs_image_score_difference']:.12g}；连续像素图最大绝对差={checks['max_abs_map_difference']:.12g}；像素计数完全一致={checks['all_pixel_counts_unchanged']}。仅改变图像阈值，不改变像素输出。",'',
      f"整图FPR Wilson95%区间：{summary['image_FPR_wilson95']}；FNR Wilson95%区间：{summary['image_FNR_wilson95']}。样本仅14正常/56异常，不把观察零错误视为总体零错误。",'',
      '**下一步保持此版本与全部结果，使用另外采集的未见正常图和缺陷图验证。** 本复测不能恢复全新独立性，也不代表对新照明、相机或工件的全面升级。WRN50相对R18的额外模型成本没有因调阈值消失。','',
      '## 文件','',
      '- protocol.json：运行前协议与冻结哈希。',
      '- wide_resnet50/predictions.json、summary.json：逐图和汇总。',
      '- wide_resnet50/maps/：本次连续分数图。',
      '- verification.json：冻结身份、前向次数、独立计数核验及旧结果对比。',
      '- 选择规则审计.json：阈值复算与规则制定时点披露。']
    md='\n'.join(lines)+'\n';(out/'WRN50_v2固定版本复测报告.md').write_text(md,encoding='utf-8')
    header='<tr><th>图片</th><th>GT</th><th>原始分数</th><th>v1预测</th><th>v2预测</th><th>FP像素</th><th>FN像素</th></tr>'
    body=''.join(f"<tr><td>{html.escape(r['image_path'])}</td><td>{r['label']}</td><td>{r['raw_score']:.12f}</td><td>{oldrows[r['image_path']]['pred_label']}</td><td>{r['pred_label']}</td><td>{r['pixel_counts']['FP']}</td><td>{r['pixel_counts']['FN']}</td></tr>" for r in rows)
    css='<style>body{max-width:1200px;margin:35px auto;padding:0 20px;font:16px/1.8 system-ui;color:#18374b;background:#f5f8fb}table{width:100%;border-collapse:collapse;background:white}td,th{padding:9px;border:1px solid #d4e0e8;text-align:left}h1,h2{color:#126080}</style>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WRN50_v2 固定版本复测</title>'+css+original.html_summary(md)+'<h2>全部70张：已使用测试子集的固定版本复测</h2><p>标签0正常、1异常。</p><table>'+header+body+'</table><p><a href="WRN50_v2固定版本复测报告.md">Markdown报告</a> · <a href="verification.json">核验</a></p></html>'
    (out/'WRN50_v2固定版本复测报告.html').write_text(page,encoding='utf-8')
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--resume',type=Path);args=parser.parse_args()
    f,m=verify()
    if not args.run:print('PREFLIGHT_OK: frozen WRN50_v2; no test reads or inference.');return
    DEST.mkdir(parents=True,exist_ok=True)
    if args.resume:
        out=args.resume.resolve();assert out.is_relative_to(DEST) and read(MARKER)['output']==out.relative_to(ROOT).as_posix()
        assert read(out/'protocol.json')['frozen_sha256']==file_sha256(FREEZE)
        if (out/'status.json').exists() and read(out/'status.json')['status']=='completed':print('Already completed. No repeat.');return
    else:
        out=DEST/('WRN50_v2_'+now().strftime('%Y%m%d_%H%M%S_%f'))
        with MARKER.open('x',encoding='utf-8') as stream:json.dump(dict(output=out.relative_to(ROOT).as_posix(),created_at=now().isoformat()),stream,ensure_ascii=False,indent=2)
        out.mkdir()
        write(out/'protocol.json',dict(version='WRN50_v2',frozen_sha256=file_sha256(FREEZE),frozen_at=f['frozen_at'],started_at=now().isoformat(),authorized_by_user=True,thresholds={'image':f['image_threshold'],'pixel':f['pixel_threshold']},checkpoint_sha256=f['checkpoint_sha256'],test_images=m['splits']['held_out'],manifest_sha256=m['manifest_sha256'],new_untouched_test=False,previous_test_results_seen=True,evaluation_kind='fixed-version reevaluation on previously used test subset',threshold_scan=False,training_runs=0,expected_model_forward_passes=70,comparators='Historical R18/v1 results only',source_sha256=file_sha256(__file__)))
        write(out/'选择规则审计.json',read(ROOT/f['selection_audit']))
    write(out/'status.json',dict(status='running',updated_at=now().isoformat()))
    print('OUTPUT',str(out),flush=True)
    import torch
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    original.FREEZE=FREEZE
    cfg=dict(checkpoint=f['checkpoint'],image_threshold=f['image_threshold'],pixel_threshold=f['pixel_threshold'])
    rows,summary=original.run_model('wide_resnet50',cfg,m,out)
    verify()
    oldr={r['image_path']:r for r in read(PRIOR/'wide_resnet50/predictions.json')}
    olds={n:read(PRIOR/n/'summary.json') for n in ['resnet18','wide_resnet50']}
    maxmap=0.
    for r in rows:
        with np.load(safe_file(out,r['map_path']),allow_pickle=False) as a,np.load(safe_file(PRIOR,oldr[r['image_path']]['map_path']),allow_pickle=False) as b:
            maxmap=max(maxmap,float(np.abs(a['anomaly_map']-b['anomaly_map']).max()))
    checks=dict(all_pixel_counts_unchanged=all(r['pixel_counts']==oldr[r['image_path']]['pixel_counts'] for r in rows),max_abs_image_score_difference=max(abs(r['raw_score']-oldr[r['image_path']]['raw_score']) for r in rows),max_abs_map_difference=maxmap)
    from sklearn.metrics import confusion_matrix
    tn,fp,fn,tp=confusion_matrix([r['label'] for r in rows],[r['pred_label'] for r in rows],labels=[0,1]).ravel()
    assert {k:int(v) for k,v in zip(['TN','FP','FN','TP'],[tn,fp,fn,tp])}==summary['image_counts']
    assert len(rows)==len({r['image_path'] for r in rows})==70
    checks.update(frozen_identities_unchanged=True,thresholds_unchanged=True,sklearn_image_confusion_matches=True,distinct_images=70,model_forward_passes=70,training_runs=0,threshold_scans=0,new_untouched_test=False)
    write(out/'summary.json',summary);write(out/'verification.json',checks)
    report(out,f,rows,summary,oldr,olds,checks)
    verify()
    write(out/'status.json',dict(status='completed',completed_at=now().isoformat(),distinct_images=70,model_forward_passes=70,training_runs=0,thresholds_unchanged=True,new_untouched_test=False))
    print('COMPLETED',str(out),flush=True)
    print(json.dumps(dict(image_counts=summary['image_counts'],pixel_metrics=summary['pixel_metrics'],checks=checks),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
