"""Manual sigma4/2 development ablation at fixed256 input and frozen bank.

No fitting. Each smoothing choice receives its own normal-only pixel threshold.
The already-recorded image q975 threshold stays fixed across both choices.
"""
import argparse
import copy
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback

import numpy as np

if __package__:
    from .analyze_thresholds import checked_inputs
    from .baseline_utils import file_sha256, load_manifest, normal_threshold, safe_file
    from .compare_resolution import calibrate, evaluate
    from .resolution_metrics import paired_localization
    from .run_trusted_baseline import load_checkpoint, load_mask, load_tensor, make_transform, state_sha256, write_json
    from .smoothing_backend import use_smoothing
else:
    from analyze_thresholds import checked_inputs
    from baseline_utils import file_sha256, load_manifest, normal_threshold, safe_file
    from compare_resolution import calibrate, evaluate
    from resolution_metrics import paired_localization
    from run_trusted_baseline import load_checkpoint, load_mask, load_tensor, make_transform, state_sha256, write_json
    from smoothing_backend import use_smoothing

SIGMAS = (4, 2)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def inspect_inputs(args):
    import torch
    identities, report, cal, dev = checked_inputs(args.baseline_dir)
    manifest = load_manifest(args.baseline_dir / 'split_manifest.json')
    policy = json.loads(args.policy.read_text(encoding='utf-8'))
    threshold = normal_threshold([r['raw_score'] for r in cal], [r['label'] for r in cal], .975)
    if (report['config']['image_size'] != [256, 256] or report['memory_bank_shape'] != [1802, 384]
            or policy['checkpoint_sha256'] != identities['model.pt']
            or policy['manifest_sha256'] != manifest['manifest_sha256']
            or policy['image_quantile'] != .975 or policy['quantile_method'] != 'linear'
            or policy['image_threshold_for_this_checkpoint'] != threshold
            or policy['pixel_threshold'] != report['thresholds']['pixel_threshold']):
        raise ValueError('Require the frozen256 baseline and its recorded q975 policy')
    if version('anomalib') != '2.2.0' or version('anomalib') != report['environment']['anomalib']:
        raise ValueError('Use the reference anomalib2.2.0 environment')
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for measured inference costs')
    source_hashes = dict(identities)
    images = masks = 0
    for rows in (cal, dev):
        for row in rows:
            record = manifest['files'][row['image_path']]
            if file_sha256(safe_file(args.data_root, row['image_path'])) != record['image_sha256']:
                raise ValueError('Input image identity changed')
            images += 1
            if record['mask_path'] is not None:
                if file_sha256(safe_file(args.data_root, record['mask_path'])) != record['mask_sha256']:
                    raise ValueError('Input mask identity changed')
                masks += 1
            source_hashes[row['map_path']] = file_sha256(safe_file(args.baseline_dir, row['map_path']))
    info = {'status': 'smoothing_preflight_passed_no_forward', 'checked_at': now(),
            'verified_images': images, 'verified_masks': masks, 'verified_baseline_maps': len(cal)+len(dev),
            'image_size': [256,256], 'sigmas': [4,2], 'pixel_quantile_each': .995,
            'fixed_image_threshold': threshold, 'checkpoint_sha256': identities['model.pt'],
            'manifest_sha256': manifest['manifest_sha256'], 'held_out_images_opened': 0,
            'new_bank_fitted': False, 'gpu': torch.cuda.get_device_name(0)}
    return source_hashes, report, manifest, cal, dev, threshold, info


def relabel_pairs(rows4, rows2):
    """Reuse validated pairing math, replacing its historical resolution labels."""
    result = paired_localization(rows4, rows2)
    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key.replace('_256', '_sigma4').replace('_512', '_sigma2'): convert(item)
                    for key,item in value.items()}
        return value
    result = convert(result)
    result['baseline'] = 'sigma4'
    result['candidate'] = 'sigma2'
    result['small_sample_limits'] = [
        'One category, seen development data; not final test or generalization evidence.',
        'GT union area per image is not individual connected-defect size.',
        'Only3 images below1% area and nonebelow0.1% in the frozen development split.']
    return result


def decision_checks(qualities, paired):
    """Conservative descriptive checks; do not silently trade recall for IoU."""
    left, right = qualities[4], qualities[2]
    group = '0.1_to_lt_1_percent'
    def ge(a,b):
        return a >= b if a is not None and b is not None else None
    def gt(a,b):
        return a > b if a is not None and b is not None else None
    checks = {
        'image_decisions_unchanged': paired['image_prediction_changes'] == 0,
        'pooled_defect_pixel_recall_not_lower': ge(right['pixel_metrics']['recall'], left['pixel_metrics']['recall']),
        'small_group_pixel_recall_not_lower': ge(right['area_groups'][group]['pixel_metrics']['recall'], left['area_groups'][group]['pixel_metrics']['recall']),
        'normal_pixel_false_positive_rate_not_higher': ge(left['normal_false_positive_pixels']['rate'],right['normal_false_positive_pixels']['rate']),
        'anomaly_macro_IoU_improved': gt(right['anomaly_macro']['pixel_IoU'],left['anomaly_macro']['pixel_IoU']),
        'small_macro_IoU_improved': gt(right['area_groups'][group]['macro']['pixel_IoU'],left['area_groups'][group]['macro']['pixel_IoU']),
    }
    return {'checks': checks, 'all_quality_conditions_met': all(value is True for value in checks.values()),
            'new_sigma_automatically_adopted': False,
            'cost_policy': 'Report measured cost ratios; quality checks alone do not adopt a configuration.'}


def compare_saved_maps(output, sigma_rows, baseline_rows, baseline_dir, split):
    originals = {r['image_path']: r for r in baseline_rows}
    if set(originals) != {r['image_path'] for r in sigma_rows}:
        raise ValueError('Baseline and repeated paths disagree')
    maximum = 0.0
    for row in sigma_rows:
        with np.load(output / row['map_path']) as new, np.load(safe_file(baseline_dir, originals[row['image_path']]['map_path'])) as old:
            a, b = new['anomaly_map'], old['anomaly_map']
            if a.shape != b.shape or not np.allclose(a,b,atol=1e-4,rtol=1e-5):
                raise ValueError(f'sigma4 reference map reproduction failed: {split}/{row["image_path"]}')
            maximum = max(maximum,float(np.abs(a-b).max()))
    return {'maps_close': True, 'max_abs_difference': maximum}


def benchmark(model, config, manifest, data_root, warmup=10, repeats=3):
    import torch
    from torch.nn import functional as F
    paths = manifest['splits']['development']
    transform = make_transform(config)
    cpu_images = [load_tensor(data_root,p,transform) for p in paths]
    rows=[]
    def forward(image,shape):
        result=model(image)
        return result.pred_score,F.interpolate(result.anomaly_map,size=shape,mode='bilinear',align_corners=False)
    with torch.inference_mode():
        for repeat in range(repeats):
            order=SIGMAS if repeat%2==0 else SIGMAS[::-1]
            for sigma in order:
                with use_smoothing(model,sigma):
                    for index in range(warmup):
                        n=index%len(paths)
                        rec=manifest['files'][paths[n]]
                        image=cpu_images[n].unsqueeze(0).to('cuda:0')
                        result=forward(image,(rec['height'],rec['width']))
                        torch.cuda.synchronize()
                        del image,result
                    for path,cpu_image in zip(paths,cpu_images):
                        rec=manifest['files'][path]
                        image=cpu_image.unsqueeze(0).to('cuda:0')
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        before=torch.cuda.memory_allocated()
                        start=time.perf_counter()
                        result=forward(image,(rec['height'],rec['width']))
                        torch.cuda.synchronize()
                        elapsed=(time.perf_counter()-start)*1000
                        peak=torch.cuda.max_memory_allocated()
                        rows.append({'sigma':sigma,'repeat':repeat,'image_path':path,'milliseconds':elapsed,
                                     'allocated_before_bytes':int(before),'peak_allocated_bytes':int(peak),'incremental_peak_bytes':int(peak-before)})
                        del result,image
                print(f'  benchmark round {repeat+1}/{repeats}: sigma={sigma}',flush=True)
    summary={}
    for sigma in SIGMAS:
        selected=[r for r in rows if r['sigma']==sigma]
        values=[r['milliseconds'] for r in selected]
        summary[sigma]={'samples':len(selected),'mean_ms':float(np.mean(values)),
                        'p95_ms':float(np.quantile(values,.95,method='linear')),
                        'max_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in selected),
                        'max_incremental_peak_bytes':max(r['incremental_peak_bytes'] for r in selected),
                        'round_mean_ms':[float(np.mean([r['milliseconds'] for r in selected if r['repeat']==n])) for n in range(repeats)]}
    return rows,summary


def visualize_pairs(output, paired, manifest, data_root, thresholds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    anomalies=[r for r in paired['per_image'] if r['label']]
    selected={r['image_path'] for r in anomalies if r['area_ratio']<.01}
    selected.update(r['image_path'] for r in sorted(anomalies,key=lambda x:x['pixel_IoU_sigma4'])[:5])
    index=[]
    for row in anomalies:
        path=row['image_path']
        if path not in selected:continue
        mask=load_mask(data_root,manifest['files'][path])
        with Image.open(safe_file(data_root,path)) as image:original=np.asarray(image)
        fig,axes=plt.subplots(2,3,figsize=(12,8))
        axes[0,0].imshow(original);axes[0,0].set_title('Original')
        axes[1,0].imshow(mask,cmap='gray',vmin=0,vmax=1);axes[1,0].set_title('Ground truth')
        for y,sigma in enumerate(SIGMAS):
            with np.load(output/f'sigma{sigma}/maps/development'/Path(path).with_suffix('.npz')) as z:a=z['anomaly_map']
            tau=thresholds[sigma]['pixel_threshold']
            axes[y,1].imshow(original)
            axes[y,1].imshow(a/max(tau,1e-12),cmap='turbo',vmin=0,vmax=2,alpha=.55)
            axes[y,1].set_title(f'sigma={sigma}: map / threshold (display only)')
            axes[y,2].imshow(a>tau,cmap='gray',vmin=0,vmax=1)
            axes[y,2].contour(mask,levels=[.5],colors=['red'],linewidths=.6)
            axes[y,2].set_title(f'sigma={sigma}: IoU={row[f"pixel_IoU_sigma{sigma}"]:.4f}, red=GT')
        for axis in axes.flat:axis.axis('off')
        fig.suptitle(f'256 input | {path} | GT area={row["area_ratio"]*100:.3f}%')
        fig.tight_layout()
        destination=output/'可视化'/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(destination,dpi=120)
        plt.close(fig)
        index.append({'image_path':path,'visualization':destination.relative_to(output).as_posix()})
    write_json(output/'visualization_index.json',index)


def run_experiment(args, inputs, output):
    import torch
    identities,baseline,manifest,old_cal,old_dev,image_threshold,info=inputs
    output.mkdir(parents=True,exist_ok=False)
    def stage(name):write_json(output/'status.json',{'status':'running','stage':name,'at':now()})
    stage('initialize')
    write_json(output/'preflight.json',info)
    write_json(output/'split_manifest.json',manifest)
    config=copy.deepcopy(baseline['config'])
    config.update({'image_size':[256,256],'image_quantile':.975,'pixel_quantile':.995})
    run_config={'status':'started_not_completed','started_at':now(),'sigmas':[4,2],'image_size':[256,256],
                'bank_reused':True,'fixed_image_threshold':image_threshold,'pixel_quantile_each':.995,
                'warmup_each_round_each_sigma':args.warmup,'repeats':args.repeats,'source_files_sha256':identities,
                'policy_sha256':file_sha256(args.policy),'manifest_sha256':manifest['manifest_sha256'],
                'source_sha256':{name:file_sha256(Path(__file__).with_name(name)) for name in
                                 ('compare_smoothing.py','smoothing_backend.py','compare_resolution.py','resolution_metrics.py','run_trusted_baseline.py','baseline_utils.py','analyze_thresholds.py')},
                'environment':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'anomalib':version('anomalib'),
                               'numpy':version('numpy'),'sklearn':version('scikit-learn'),'gpu':torch.cuda.get_device_name(0)}}
    write_json(output/'config.json',run_config)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model,saved=load_checkpoint(args.baseline_dir/'model.pt',torch.device('cuda:0'))
    model.requires_grad_(False)
    if saved['model_config']!=baseline['config'] or saved['manifest_sha256']!=manifest['manifest_sha256']:
        raise ValueError('Loaded checkpoint identity mismatch')
    if state_sha256(model.feature_extractor.state_dict())!=baseline['frozen_backbone_state_sha256']:
        raise ValueError('Loaded backbone identity mismatch')
    original_state=state_sha256(model.state_dict())
    original_generator=model.anomaly_map_generator
    thresholds,cal_rows,rows,qualities={},{},{},{}
    reproduction={}
    print('[1/3] CALIBRATE: same44 normal images, sigma4 / sigma2 independently',flush=True)
    stage('calibrate')
    old_cal_scores={r['image_path']:r['raw_score'] for r in old_cal}
    for sigma in SIGMAS:
        destination=output/f'sigma{sigma}'
        destination.mkdir()
        write_json(destination/'config.json',{**config,'smoothing_sigma':sigma,'kernel_size':2*int(4*sigma+.5)+1})
        with use_smoothing(model,sigma):
            thresholds[sigma],cal_rows[sigma]=calibrate(model,config,manifest,args.data_root,destination)
        if not all(np.isclose(r['raw_score'],old_cal_scores[r['image_path']],atol=1e-6,rtol=1e-6) for r in cal_rows[sigma]):
            raise ValueError('Smoothing unexpectedly changed calibration image scores')
        computed=thresholds[sigma]['image_threshold']
        if not np.isclose(computed,image_threshold,atol=1e-6,rtol=1e-6):
            raise ValueError('Calibration image threshold differs from frozen policy')
        thresholds[sigma].update({'image_threshold_recomputed_for_check':computed,'image_threshold':image_threshold,
                                  'image_threshold_policy':'fixed previously recorded q975 rule; not retuned in this ablation'})
        write_json(destination/'calibration.json',thresholds[sigma])
    if not np.isclose(thresholds[4]['pixel_threshold'],baseline['thresholds']['pixel_threshold'],atol=1e-4,rtol=1e-5):
        raise ValueError('sigma4 pixel calibration cannot reproduce baseline')
    reproduction['normal_maps']=compare_saved_maps(output/'sigma4',cal_rows[4],old_cal,args.baseline_dir,'normal_calibration')
    print('[2/3] EVALUATE: fixed45 development images, original-coordinate localization',flush=True)
    stage('evaluate')
    old_dev_scores={r['image_path']:r['raw_score'] for r in old_dev}
    for sigma in SIGMAS:
        print(f'  evaluate sigma={sigma}',flush=True)
        with use_smoothing(model,sigma):
            rows[sigma],qualities[sigma]=evaluate(model,config,thresholds[sigma],manifest,args.data_root,output/f'sigma{sigma}')
        if not all(np.isclose(r['raw_score'],old_dev_scores[r['image_path']],atol=1e-6,rtol=1e-6) for r in rows[sigma]):
            raise ValueError('Smoothing unexpectedly changed development image scores')
    reproduction['development_maps']=compare_saved_maps(output/'sigma4',rows[4],old_dev,args.baseline_dir,'development')
    paired=relabel_pairs(rows[4],rows[2])
    if paired['image_prediction_changes']:
        raise ValueError('Image predictions changed despite fixed scoring and threshold')
    write_json(output/'paired_localization.json',paired)
    checks=decision_checks(qualities,paired)
    print('[3/3] BENCHMARK: same256 model, sigma order4/2 then2/4 then4/2',flush=True)
    stage('benchmark')
    timing_rows,performance=benchmark(model,config,manifest,args.data_root,args.warmup,args.repeats)
    write_json(output/'timings.json',timing_rows)
    stage('visualize_and_verify')
    visualize_pairs(output,paired,manifest,args.data_root,thresholds)
    if (model.anomaly_map_generator is not original_generator or state_sha256(model.state_dict())!=original_state
            or any(file_sha256(safe_file(args.baseline_dir,name))!=digest for name,digest in identities.items())
            or file_sha256(args.policy)!=run_config['policy_sha256']):
        raise ValueError('Frozen model, baseline evidence or policy changed')
    run_config.update({'status':'completed','completed_at':now()})
    write_json(output/'config.json',run_config)
    cost={'sigma2_over_sigma4_mean_time':performance[2]['mean_ms']/performance[4]['mean_ms'],
          'sigma2_over_sigma4_peak_allocated':performance[2]['max_peak_allocated_bytes']/performance[4]['max_peak_allocated_bytes']}
    report={'status':'completed_development_smoothing_comparison_not_final_test','completed_at':now(),
            'config':run_config,'qualities':{f'sigma{k}':v for k,v in qualities.items()},
            'thresholds':{f'sigma{k}':v for k,v in thresholds.items()},'baseline_reproduction':reproduction,
            'decision':checks,'performance':{f'sigma{k}':v for k,v in performance.items()},'cost_ratios':cost,
            'development_images':len(manifest['splits']['development']),'held_out_images_opened':0,
            'new_model_or_bank_fitted':False,'image_scores_and_decisions_unchanged':True,
            'model_state_unchanged':True,'timing_boundary':'batch1 CUDA-ready input through frozen full model and original-size GPU interpolation; synchronized wall time; excludes decoding/preprocessing/H2D/D2H/metrics/writing/generator setup',
            'memory_boundary':'PyTorch peak allocated bytes with one model and active generator; inactive generator onCPU; includes model/input/output, not total device VRAM',
            'limitations':['Only3 sub1% GT-union-area images and nonebelow0.1%; singlecategory and checkpoint.',
                           'Post-hoc development hypothesis after resolution analysis; not final-test evidence.',
                           'Each sigma recalibrates its pixel threshold with the same normal-only rule.',
                           'No assumed sharper boundary or recall improvement; no automatic adoption.']}
    write_json(output/'report.json',report)
    def metric(value):return 'N/A' if value is None else f'{value:.4f}'
    lines=['256输入：sigma4 / sigma2 平滑强度对照已完成（开发集）','原模型和1802条特征库复用；图像阈值固定，像素阈值分别正常校准。']
    group='0.1_to_lt_1_percent'
    for sigma in SIGMAS:
        q,perf=qualities[sigma],performance[sigma]
        small=q['area_groups'][group]
        lines.append(f'sigma={sigma}: 全部异常图平均IoU={metric(q["anomaly_macro"]["pixel_IoU"])}，F1={metric(q["anomaly_macro"]["pixel_F1"])}，AP={metric(q["anomaly_macro"]["pixel_AP"])}')
        lines.append(f'  总体缺陷像素召回={metric(q["pixel_metrics"]["recall"])}，正常图像素误报率={metric(q["normal_false_positive_pixels"]["rate"])}')
        lines.append(f'  小面积组n={small["n"]}：平均IoU={metric(small["macro"]["pixel_IoU"])}，像素召回={metric(small["pixel_metrics"]["recall"])}')
        lines.append(f'  平均={perf["mean_ms"]:.3f}ms，P95={perf["p95_ms"]:.3f}ms，分配显存峰值={perf["max_peak_allocated_bytes"]/1024**2:.3f}MiB')
    lines += [f'定位改善且不降低召回/不增加正常像素误报的全部条件满足：{checks["all_quality_conditions_met"]}',
              f'条件明细：{checks["checks"]}',f'sigma2/sigma4：平均耗时比={cost["sigma2_over_sigma4_mean_time"]:.3f}，分配显存峰值比={cost["sigma2_over_sigma4_peak_allocated"]:.3f}',
              '图像判断保持不变；没有自动采用新sigma。请结合可视化和逐图差异分析。',
              '指标不改善或召回下降也是有效发现；小面积组仅3张，保留测试未使用。']
    (output/'结果摘要.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(output/'status.json',{'status':'completed','completed_at':now()})
    print('\n'.join(lines),flush=True)
    print(f'SMOOTHING_COMPARISON_COMPLETED | {output}',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if min(args.warmup,args.repeats)<1:parser.error('Warmup/repeats must be positive')
    args.baseline_dir,args.data_root,args.policy=args.baseline_dir.resolve(),args.data_root.resolve(),args.policy.resolve()
    inputs=inspect_inputs(args)
    if args.check_only:
        print(json.dumps(inputs[-1],ensure_ascii=False,indent=2),flush=True)
        print('SMOOTHING_PREFLIGHT_PASSED | no fitting or model forward',flush=True)
        return
    if args.output_root is None:parser.error('--output-root is required')
    root=args.output_root.resolve()
    if any(root==p or root.is_relative_to(p) for p in (args.baseline_dir,args.data_root)):
        parser.error('Output must be outside frozen baseline/raw data')
    output=root/('smoothing_comparison_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f'))
    print(f'OUTPUT: {output}',flush=True)
    try:run_experiment(args,inputs,output)
    except Exception as error:
        if output.exists():
            write_json(output/'status.json',{'status':'failed_not_completed','at':now(),'error_type':type(error).__name__,'error':str(error)})
            (output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':main()
