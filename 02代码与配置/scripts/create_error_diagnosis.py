"""Render existing predictions into a local error report; no model forward."""
import argparse
from datetime import datetime, timedelta, timezone
import html
from html.parser import HTMLParser
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

if __package__:
    from .baseline_utils import binary_counts, file_sha256, load_manifest, safe_file, summarize_counts
else:
    from baseline_utils import binary_counts, file_sha256, load_manifest, safe_file, summarize_counts

CASES = ('metal_nut/test/bent/002.png','metal_nut/test/color/000.png','metal_nut/test/color/001.png',
         'metal_nut/test/scratch/000.png','metal_nut/test/scratch/003.png')
LABELS = {'256_sigma4':'256 / sigma4','256_sigma2':'256 / sigma2','512_sigma4':'512 / sigma4'}


def save_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def error_colors(original, gt, prediction):
    result=original.astype(np.float32)/255*.22
    result[gt & prediction]=[.15,.85,.35]
    result[~gt & prediction]=[1.,.15,.15]
    result[gt & ~prediction]=[.12,.35,1.]
    return result


def render_case(output, number, relative, original, gt, variants, arrays, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y,x=np.where(gt)
    crop=(slice(max(0,int(y.min())-20),min(gt.shape[0],int(y.max())+21)),
          slice(max(0,int(x.min())-20),min(gt.shape[1],int(x.max())+21)))
    files={}
    for kind,section in (('full',(slice(None),slice(None))),('zoom',crop)):
        fig,axes=plt.subplots(3,4,figsize=(16,11.5))
        for index,(key,variant) in enumerate(variants.items()):
            scores=arrays[key]
            threshold=variant['calibration']['pixel_threshold']
            pred=scores>threshold
            row=results[key]
            axes[index,0].imshow(original[section])
            axes[index,0].contour(gt[section],levels=[.5],colors=['yellow'],linewidths=.7)
            axes[index,0].set_title(f'{LABELS[key]} | original + GT outline')
            axes[index,1].imshow(original[section])
            axes[index,1].imshow((scores/threshold)[section],cmap='turbo',alpha=.6,vmin=0,vmax=2)
            axes[index,1].set_title('map / own pixel threshold (display only)')
            axes[index,2].imshow(pred[section],cmap='gray',vmin=0,vmax=1)
            axes[index,2].contour(gt[section],levels=[.5],colors=['yellow'],linewidths=.7)
            axes[index,2].set_title(f'mask | IoU={row["IoU"]:.4f}, recall={row["recall"]:.4f}')
            axes[index,3].imshow(error_colors(original,gt,pred)[section])
            axes[index,3].set_title(f'FP(red)={row["FP"]}, FN(blue)={row["FN"]}')
        for axis in axes.flat:axis.axis('off')
        fig.suptitle(f'{relative} | GREEN=correct defect, RED=extra normal area, BLUE=missed defect\nNumbers are for the full image; zoom uses GT bounding box +20 original pixels.',fontsize=12)
        fig.tight_layout()
        name=f'case_{number:02d}_{Path(relative).parent.name}_{Path(relative).stem}_{kind}.png'
        destination=output/'图像'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(destination,dpi=120)
        plt.close(fig)
        files[kind]=destination.relative_to(output).as_posix()
    return files


def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoothing-run',type=Path,default=root/'04实验结果/06平滑强度对照/smoothing_comparison_20261023_202433_801686')
    parser.add_argument('--resolution-run',type=Path,default=root/'04实验结果/05分辨率对照/resolution_comparison_20261023_161820_426134')
    parser.add_argument('--data-root',type=Path,default=root/'03原始数据/解压数据/MVTecAD')
    parser.add_argument('--output-root',type=Path,default=root/'04实验结果/07错误诊断')
    args=parser.parse_args()
    smooth_report=json.loads((args.smoothing_run/'report.json').read_text(encoding='utf-8'))
    resolution_report=json.loads((args.resolution_run/'report.json').read_text(encoding='utf-8'))
    if (smooth_report['status']!='completed_development_smoothing_comparison_not_final_test'
            or resolution_report['status']!='completed_development_resolution_comparison_not_final_test'
            or smooth_report['held_out_images_opened']!=0 or resolution_report['held_out_images_opened']!=0):
        raise ValueError('Completed development reports without held-out inference are required')
    manifest=load_manifest(args.smoothing_run/'split_manifest.json')
    other_manifest=load_manifest(args.resolution_run/'split_manifest.json')
    if manifest['manifest_sha256']!=other_manifest['manifest_sha256']:
        raise ValueError('Sources use different frozen splits')
    variants={}
    source_hashes={}
    for run in (args.smoothing_run,args.resolution_run):
        for name in ('report.json','split_manifest.json'):
            path=run/name;source_hashes[path]=file_sha256(path)
    for key,folder,quality in (
        ('256_sigma4',args.smoothing_run/'sigma4',smooth_report['qualities']['sigma4']),
        ('256_sigma2',args.smoothing_run/'sigma2',smooth_report['qualities']['sigma2']),
        ('512_sigma4',args.resolution_run/'512',resolution_report['qualities']['512'])):
        rows=json.loads((folder/'development_predictions.json').read_text(encoding='utf-8'))
        calibration=json.loads((folder/'calibration.json').read_text(encoding='utf-8'))
        if {r['image_path'] for r in rows}!=set(manifest['splits']['development']) or len(rows)!=len(manifest['splits']['development']):
            raise ValueError('Development memberships differ')
        variants[key]={'folder':folder,'rows':{r['image_path']:r for r in rows},'calibration':calibration,'quality':quality}
        for name in ('development_predictions.json','calibration.json'):
            path=folder/name;source_hashes[path]=file_sha256(path)
    output_root=args.output_root.resolve()
    for protected in (args.data_root,args.smoothing_run,args.resolution_run):
        if output_root==protected.resolve() or output_root.is_relative_to(protected.resolve()):
            raise ValueError('Write diagnosis outside existing sources')
    output=output_root/('error_diagnosis_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    source_rows=[];selected={};totals={key:{k:0 for k in ('TP','FP','FN','TN')} for key in variants}
    for relative in manifest['splits']['development']:
        record=manifest['files'][relative]
        if record['mask_path'] is None:gt=np.zeros((record['height'],record['width']),dtype=bool)
        else:
            mask_file=safe_file(args.data_root,record['mask_path'])
            if file_sha256(mask_file)!=record['mask_sha256']:raise ValueError('Mask identity changed')
            with Image.open(mask_file) as image:gt=np.asarray(image)>0
        arrays={};per_variant={}
        for key,variant in variants.items():
            row=variant['rows'][relative]
            if row['label']!=record['label'] or row['image_sha256']!=record['image_sha256'] or row['mask_sha256']!=record['mask_sha256']:
                raise ValueError('Prediction identity changed')
            path=safe_file(variant['folder'],row['map_path'])
            source_hashes[path]=file_sha256(path)
            with np.load(path) as data:score_map=data['anomaly_map']
            if score_map.shape!=gt.shape or not np.isfinite(score_map).all():raise ValueError('Map shape/values invalid')
            counts=binary_counts(gt,score_map,variant['calibration']['pixel_threshold'])
            if counts!=row['pixel_counts']:raise ValueError('Recomputed pixel counts differ from saved results')
            for k in counts:totals[key][k]+=counts[k]
            per_variant[key]={**summarize_counts(counts),'image_score':row['raw_score'],
                              'image_threshold':variant['calibration']['image_threshold'],'pred_label':row['pred_label'],
                              'image_score_margin':row['raw_score']-variant['calibration']['image_threshold'],
                              'pixel_AP':row['pixel_AP']}
            arrays[key]=score_map
        source_rows.append({'image_path':relative,'label':record['label'],'area_ratio':float(gt.mean()),'variants':per_variant})
        if relative in CASES:
            image_file=safe_file(args.data_root,relative)
            if file_sha256(image_file)!=record['image_sha256']:raise ValueError('Selected original image changed')
            with Image.open(image_file) as image:original=np.asarray(image.convert('RGB'))
            matrix=[]
            for map_key in ('256_sigma4','256_sigma2'):
                for tau_key in ('256_sigma4','256_sigma2'):
                    tau=variants[tau_key]['calibration']['pixel_threshold']
                    matrix.append({'map':map_key,'pixel_threshold_from':tau_key,'pixel_threshold':tau,
                                   **summarize_counts(binary_counts(gt,arrays[map_key],tau))})
            pred4=arrays['256_sigma4']>variants['256_sigma4']['calibration']['pixel_threshold']
            pred2=arrays['256_sigma2']>variants['256_sigma2']['calibration']['pixel_threshold']
            lost=gt&pred4&~pred2;recovered=gt&~pred4&pred2
            distance=distance_transform_edt(gt)
            selected[relative]={'results':per_variant,'threshold_diagnostic_only':matrix,
                                'newly_missed_pixels':int(lost.sum()),'recovered_pixels':int(recovered.sum()),
                                'new_misses_within_5_original_pixels_of_GT_boundary':int((lost&(distance<=5)).sum()),
                                'new_misses_more_than_5_original_pixels_inside_GT':int((lost&(distance>5)).sum()),
                                'files':render_case(output,CASES.index(relative)+1,relative,original,gt,variants,arrays,per_variant)}
    for key,counts in totals.items():
        if counts!=variants[key]['quality']['pixel_counts']:raise ValueError('Global counts differ from original reports')
    for path,digest in source_hashes.items():
        if file_sha256(path)!=digest:raise ValueError('Source artifact changed during diagnosis')
    notes={
        CASES[0]:'sigma2减少多圈区域且本图没有新增漏分；512进一步减少FP，但新增581个缺陷像素漏分。',
        CASES[1]:'两种256方案均覆盖全部GT，sigma2只减少602个FP；512的FP更少，但仍有明显外扩。',
        CASES[2]:'sigma2减少1164个FP，同时新增274个FN，IoU反而下降。这张图不能用“平滑越小越准确”解释。',
        CASES[3]:'sigma4漏分2791个像素；sigma2配套校准后漏分4039个，净增1248。固定原像素阈值时只多漏96个，再换新阈值后多漏1152个。新漏掉的1337个像素中859个距GT边界超过5原图像素，不只是边缘收缩；同时找回89个原本漏分像素。',
        CASES[4]:'512定位IoU约0.5366，优于256约0.4304，但图像分数14.639861低于自身阈值14.654193，故图像判正常。图像加权评分和像素异常图是分别计算的，定位好不保证整图判异常。具体是最大patch距离还是邻域权重导致分数降低，保存的最终输出不足以区分。'}
    findings={'status':'completed_saved_output_error_diagnosis','created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'manifest_sha256':manifest['manifest_sha256'],'development_images':len(source_rows),
              'selected_cases':selected,'all_image_records':source_rows,
              'verified_global_pixel_counts':totals,'notes':notes,
              'source_sha256':{path.relative_to(root).as_posix():digest for path,digest in source_hashes.items()},
              'model_forward_run':False,'new_model_fitted':False,'held_out_images_opened':0,'sources_unchanged':True,
              'format_note':'Read existing PNG images/masks and NPZ float maps; generate new scientific PNG panels, HTML, Markdown and JSON. Original artifacts unchanged.',
              'limits':['Threshold-crossing table is a post-hoc diagnostic, not a newly calibrated or adopted policy.',
                        'Five-pixel GT-boundary band is a descriptive original-coordinate definition, not physical defect size.',
                        'Sigma2 and its independently calibrated pixel threshold change together under the chosen rule.',
                        'Single seen-development category; small-area group contains only3 cases.']}
    save_json(output/'逐图诊断.json',findings)
    legend='<p class="legend"><b class="green">绿色：正确覆盖的缺陷</b>　<b class="red">红色：多圈出的正常区域（FP）</b>　<b class="blue">蓝色：漏掉的真实缺陷（FN）</b>。黄色细线是GT轮廓。</p>'
    cards=[]
    markdown=['# 错误诊断报告','', '当前仍保留256＋sigma4。报告仅整理已有输出，未重新推理或建库，原始图片和NPZ保持原样。',
              '', 'sigma较大可能扩散/连接弱响应，但减小sigma后还会改变正常校准阈值。残留外扩不能仅凭本次对照归因于sigma。', '',
              '红色=FP多圈正常区，蓝色=FN漏掉缺陷，绿色=TP正确缺陷。所有数值都对应整张原图。','']
    for i,relative in enumerate(CASES,1):
        case=selected[relative]
        body=''
        for key,row in case['results'].items():
            body+=f'<tr><td>{LABELS[key]}</td><td>{row["IoU"]:.4f}</td><td>{row["recall"]*100:.2f}%</td><td>{row["FP"]}</td><td>{row["FN"]}</td><td>{row["image_score"]:.4f} / {row["image_threshold"]:.4f}</td><td>{"异常" if row["pred_label"] else "正常"}</td></tr>'
        diag=''
        for row in case['threshold_diagnostic_only']:
            diag+=f'<tr><td>{LABELS[row["map"]]}</td><td>{row["pixel_threshold"]:.6f}</td><td>{row["FP"]}</td><td>{row["FN"]}</td><td>{row["IoU"]:.4f}</td></tr>'
        cards.append(f'<section id="case{i}"><h2>{i}. {html.escape(relative)}</h2><p>{notes[relative]}</p><div class="scroll"><table><tr><th>配置</th><th>IoU</th><th>缺陷像素召回</th><th>FP</th><th>FN</th><th>图像分数 / 阈值</th><th>图像判断</th></tr>{body}</table></div>{legend}<a href="{case["files"]["zoom"]}"><img loading="lazy" src="{case["files"]["zoom"]}" alt="局部红蓝诊断图"></a><details><summary>查看整图与阈值影响诊断</summary><a href="{case["files"]["full"]}"><img loading="lazy" src="{case["files"]["full"]}" alt="整图红蓝诊断图"></a><p>下面只复算已有分数图，帮助区分异常图变化和阈值上移；不代表重新选定了部署阈值。</p><table><tr><th>异常图来自</th><th>用于诊断的像素阈值</th><th>FP</th><th>FN</th><th>IoU</th></tr>{diag}</table></details></section>')
        markdown.extend([f'## {relative}', '',notes[relative],'',f'![局部错误图]({case["files"]["zoom"]})',''])
    all_rows=''
    for row in source_rows:
        if not row['label']:continue
        cells=''.join(f'<td>{row["variants"][k]["IoU"]:.4f} / {row["variants"][k]["FN"]}</td>' for k in variants)
        all_rows+=f'<tr><td>{row["image_path"]}</td>{cells}</tr>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>错误诊断报告</title><style>body{margin:0;background:#f3f6fa;color:#213445;font:16px/1.7 "Microsoft YaHei","Segoe UI",sans-serif}main{max-width:1320px;margin:28px auto;padding:0 22px}section{background:white;border:1px solid #dae3ed;border-radius:12px;padding:22px;margin:20px 0}h1{font-size:30px}h2{font-size:21px}a{color:#196bb0}nav{display:flex;gap:15px;flex-wrap:wrap}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e5ebf2}img{width:100%;height:auto}.scroll{overflow:auto}.red{color:#d32d2d}.blue{color:#245dd2}.green{color:#147d37}.legend{font-size:14px}summary{cursor:pointer;color:#1764a3}footer{font-size:14px;color:#5c7183}</style><main><h1>错误诊断：到底多圈了哪里、漏掉了哪里</h1><p>当前保留256＋sigma4。本页读取已完成实验的输出，不重新建库或运行模型；3个小面积案例＋2个划痕案例。</p><p>原始PNG图片/标注和NPZ异常图无需你手动转换。新生成的PNG用于看图，JSON用于数值复核；原文件保持原样。</p>NAV<section><h2>先看结论</h2><p>较大sigma可能扩散响应，但不能据此把外扩全部归因于sigma。scratch/000的退化还与重新校准后像素阈值升高有关；scratch/003在512下属于图像级加权分数低于阈值，尽管其像素定位改善。</p><p>阈值交叉诊断不是自动调参；当前保留原有配置。下面逐图展示事实、解释及证据边界。</p></section>CARDS<section><details><summary>全部37张异常图的IoU / FN像素数</summary><table><tr><th>图片</th><th>256 / sigma4</th><th>256 / sigma2</th><th>512 / sigma4</th></tr>ALL</table></details><p><a href="逐图诊断.json">完整数值与来源哈希</a>　<a href="诊断说明.md">Markdown说明</a></p></section><footer>只使用已见开发数据，70张保留测试未读取。单类、单种子、3个小面积案例，不能推断工业泛化。图像级加权评分的底层中间量未保存，根因分析不超出已有证据。</footer></main></html>'''
    nav='<nav>'+''.join(f'<a href="#case{i}">{Path(p).parent.name}/{Path(p).name}</a>' for i,p in enumerate(CASES,1))+'</nav>'
    page=page.replace('NAV',nav).replace('CARDS',''.join(cards)).replace('ALL',all_rows)
    (output/'诊断报告.html').write_text(page,encoding='utf-8')
    (output/'诊断说明.md').write_text('\n'.join(markdown),encoding='utf-8')
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('href','src') and not value.startswith('#'):
                    if not (output/value).exists():raise ValueError(f'Missing report target: {value}')
    Links().feed(page)
    save_json(output/'status.json',{'status':'completed','selected_cases':5,'figures':10,'source_files_unchanged':True,
                                   'saved_pixel_counts_reproduced_for_all45_images_each_config':True,'html_links_verified':True})
    print(f'ERROR_DIAGNOSIS_COMPLETED | {output}',flush=True)


if __name__=='__main__':main()
