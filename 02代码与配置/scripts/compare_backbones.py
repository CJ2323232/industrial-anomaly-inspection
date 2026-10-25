"""Prespecified ResNet18 / WideResNet50 comparison on the frozen development split."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

if __package__:
    from .backbone_backend import MODEL_NAME, WEIGHT_NAME, load_pretrained_features, verify_weight_file
    from .baseline_utils import file_sha256, safe_file, normal_threshold
    from .compare_resolution import inspect_inputs, seed_all, select_bank, calibrate, evaluate
    from .compare_smoothing import compare_saved_maps
    from .comparison_reporting import visualize, write_html
    from .resolution_metrics import paired_localization
    from .run_trusted_baseline import load_checkpoint, make_model, make_transform, load_tensor, predict, state_sha256, write_json
else:
    from backbone_backend import MODEL_NAME, WEIGHT_NAME, load_pretrained_features, verify_weight_file
    from baseline_utils import file_sha256, safe_file, normal_threshold
    from compare_resolution import inspect_inputs, seed_all, select_bank, calibrate, evaluate
    from compare_smoothing import compare_saved_maps
    from comparison_reporting import visualize, write_html
    from resolution_metrics import paired_localization
    from run_trusted_baseline import load_checkpoint, make_model, make_transform, load_tensor, predict, state_sha256, write_json

NAMES = ('resnet18', 'wide_resnet50')
PLAN = {'image_size': [256, 256], 'layers': ['layer2', 'layer3'], 'num_neighbors': 9,
        'sigma': 4, 'bank_vectors_each': 1802, 'candidate_vectors_each': 180224,
        'reference_feature_dim': 384, 'candidate_feature_dim': 1536,
        'candidate_backbone': MODEL_NAME, 'fit_batch_size_candidate': 1,
        'coreset_seed': 0, 'image_quantile': .975, 'pixel_quantile': .995,
        'quantile_method': 'linear', 'decision_rule': 'strict greater-than',
        'no_candidate_prefilter': True, 'automatic_adoption': False}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def preflight(args):
    inputs = inspect_inputs(args)
    identities, baseline, manifest, cal_rows, dev_rows, old_info = inputs
    policy = read_json(args.policy)
    tau = normal_threshold([r['raw_score'] for r in cal_rows], [r['label'] for r in cal_rows], .975)
    if (policy['checkpoint_sha256'] != identities['model.pt'] or policy['manifest_sha256'] != manifest['manifest_sha256']
            or policy['image_quantile'] != .975 or policy['image_threshold_for_this_checkpoint'] != tau):
        raise ValueError('Require the recorded q975 baseline policy')
    metadata = read_json(args.weight_metadata)
    weights = verify_weight_file(args.weights, metadata['sha256'])
    for row in cal_rows + dev_rows:
        identities[row['map_path']] = file_sha256(safe_file(args.baseline_dir, row['map_path']))
    info = {'status': 'backbone_preflight_passed_no_forward', 'checked_at': now(), 'plan': PLAN,
            'verified_images': old_info['verified_images'], 'verified_masks': old_info['verified_masks'],
            'held_out_images_opened': 0, 'manifest_sha256': manifest['manifest_sha256'],
            'checkpoint_sha256': identities['model.pt'], 'weights': weights,
            'policy_sha256': file_sha256(args.policy), 'weight_metadata_sha256': file_sha256(args.weight_metadata),
            'fixed_image_threshold_reference': tau, 'gpu': torch.cuda.get_device_name(0)}
    return inputs, info


def build_bank(model, config, manifest, data_root, output):
    model.to('cuda:0').requires_grad_(False)
    model.train()
    model.feature_extractor.eval()
    pool = torch.empty((180224, 1536), dtype=torch.float32, device='cpu')
    transform = make_transform(config)
    offset = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for index, relative in enumerate(manifest['splits']['fit']):
            image = load_tensor(data_root, relative, transform).unsqueeze(0).to('cuda:0')
            embedding = model(image)
            if tuple(embedding.shape) != (1024, 1536):
                raise ValueError('Unexpected WideResNet50 feature geometry')
            pool[offset:offset + 1024].copy_(embedding.cpu())
            offset += 1024
            model.embedding_store.clear()
            del embedding, image
            if (index + 1) % 22 == 0:
                print(f'  normal wide-resnet features {index+1}/176', flush=True)
        if offset != len(pool):
            raise ValueError('Normal candidate count mismatch')
        candidates = pool.to('cuda:0')
        del pool
        bank, indices, metadata = select_bank(candidates, 1802, seed=0)
        model.memory_bank = bank
        del candidates
    model.eval()
    torch.cuda.synchronize()
    metadata.update({'seconds': time.perf_counter()-started, 'candidate_feature_count': offset,
                     'bank_shape': list(bank.shape), 'bank_bytes': bank.numel()*bank.element_size(),
                     'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                     'duration_boundary': 'reads, preprocessing, feature extraction, staging/transfers and coreset; excludes calibration/evaluation',
                     'peak_boundary': 'PyTorch GPU allocation during build, including model; CPU pool excluded'})
    write_json(output / 'coreset_indices.json', {'indices': indices, 'fit_paths_in_order': manifest['splits']['fit'],
                                               'flattening': 'image order, row-major32x32 features'})
    metadata['coreset_indices_sha256'] = file_sha256(output / 'coreset_indices.json')
    write_json(output / 'build_report.json', metadata)
    return metadata


def benchmark(models, config, manifest, data_root, warmup, repeats):
    paths = manifest['splits']['development']
    cpu_images = [load_tensor(data_root, p, make_transform(config)) for p in paths]
    def forward(model, image, size):
        prediction = model(image)
        return prediction.pred_score, F.interpolate(prediction.anomaly_map, size=size, mode='bilinear', align_corners=False)
    rows = []
    with torch.inference_mode():
        for repeat in range(repeats):
            for name in NAMES if repeat % 2 == 0 else NAMES[::-1]:
                for other in models.values():
                    other.to('cpu')
                torch.cuda.empty_cache()
                model = models[name].to('cuda:0').eval()
                for n in range(warmup):
                    index = n % len(paths)
                    rec = manifest['files'][paths[index]]
                    image = cpu_images[index].unsqueeze(0).to('cuda:0')
                    result = forward(model, image, (rec['height'], rec['width']))
                    torch.cuda.synchronize()
                    del image, result
                for path, cpu_image in zip(paths, cpu_images):
                    rec = manifest['files'][path]
                    image = cpu_image.unsqueeze(0).to('cuda:0')
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    before = torch.cuda.memory_allocated()
                    started = time.perf_counter()
                    result = forward(model, image, (rec['height'], rec['width']))
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter()-started)*1000
                    rows.append({'method': name, 'repeat': repeat, 'image_path': path, 'milliseconds': elapsed,
                                 'allocated_before_bytes': int(before), 'peak_allocated_bytes': int(torch.cuda.max_memory_allocated())})
                    del image, result
                model.to('cpu')
                print(f'  benchmark round {repeat+1}/{repeats}: {name}', flush=True)
    summary = {}
    for name in NAMES:
        selected = [row for row in rows if row['method'] == name]
        summary[name] = {'samples': len(selected), 'mean_ms': float(np.mean([r['milliseconds'] for r in selected])),
                         'p95_ms': float(np.quantile([r['milliseconds'] for r in selected], .95)),
                         'max_peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in selected),
                         'round_mean_ms': [float(np.mean([r['milliseconds'] for r in selected if r['repeat']==n])) for n in range(repeats)]}
    return rows, summary


def relabel_pair(value):
    if isinstance(value, list):
        return [relabel_pair(v) for v in value]
    if isinstance(value, dict):
        return {k.replace('_256', '_resnet18').replace('_512', '_wide_resnet50'): relabel_pair(v) for k,v in value.items()}
    return value


def run(args, inputs, info, output):
    identities, baseline, manifest, old_cal, old_dev, _ = inputs
    output.mkdir(parents=True, exist_ok=False)
    def stage(name):
        write_json(output / 'status.json', {'status': 'running', 'stage': name, 'at': now()})
    stage('initialize')
    write_json(output / 'preflight.json', info)
    write_json(output / 'split_manifest.json', manifest)
    config = {'status': 'started_not_completed', 'started_at': now(), 'plan': PLAN,
              'manifest_sha256': manifest['manifest_sha256'], 'baseline_sources_sha256': identities,
              'policy_sha256': info['policy_sha256'], 'weights': info['weights'],
              'weight_metadata_sha256': info['weight_metadata_sha256'],
              'source_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in
                                ('compare_backbones.py', 'backbone_backend.py', 'comparison_reporting.py',
                                 'compare_resolution.py', 'compare_smoothing.py', 'resolution_metrics.py',
                                 'run_trusted_baseline.py', 'baseline_utils.py', 'analyze_thresholds.py')},
              'warmup_each_round': args.warmup, 'repeats': args.repeats,
              'environment': {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'anomalib': version('anomalib'),
                              'timm': version('timm'), 'numpy': version('numpy'), 'sklearn': version('scikit-learn'),
                              'gpu': torch.cuda.get_device_name(0)}}
    write_json(output / 'config.json', config)
    seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    reference, saved = load_checkpoint(args.baseline_dir / 'model.pt', torch.device('cpu'))
    reference.requires_grad_(False)
    if (saved['manifest_sha256'] != manifest['manifest_sha256'] or
            state_sha256(reference.feature_extractor.state_dict()) != baseline['frozen_backbone_state_sha256']):
        raise ValueError('Baseline checkpoint identity mismatch')
    cfg = copy.deepcopy(baseline['config'])
    cfg.update({'image_quantile': .975, 'pixel_quantile': .995})
    candidate_cfg = copy.deepcopy(cfg)
    candidate_cfg.update({'backbone': MODEL_NAME, 'pretrained_source': info['weights']['url'], 'fit_batch_size': 1})
    candidate = make_model(candidate_cfg, pretrained=False).requires_grad_(False)
    weight_mapping = load_pretrained_features(candidate, args.weights)
    before = state_sha256(candidate.feature_extractor.state_dict())
    models = {'resnet18': reference, 'wide_resnet50': candidate}
    configs = {'resnet18': cfg, 'wide_resnet50': candidate_cfg}
    roots = {name: output / name for name in NAMES}
    for name in NAMES:
        roots[name].mkdir()
        write_json(roots[name] / 'config.json', configs[name])
    print('[1/5] FIT: WideResNet50, 176 normal images, full180224 candidates ->1802 bank', flush=True)
    stage('fit_wide_resnet50')
    build = build_bank(candidate, candidate_cfg, manifest, args.data_root, roots['wide_resnet50'])
    if state_sha256(candidate.feature_extractor.state_dict()) != before:
        raise ValueError('Pretrained backbone parameters or statistics changed')
    candidate.to('cpu')
    thresholds, rows, qualities = {}, {}, {}
    reproduction = {}
    checkpoint_checks = {}
    sizes = {}
    for name in NAMES:
        print(f'[2/5] CALIBRATE {name}: 44 normal images only', flush=True)
        stage('calibrate_' + name)
        model = models[name].to('cuda:0').eval()
        thresholds[name], calibration_rows = calibrate(model, configs[name], manifest, args.data_root, roots[name])
        if name == 'resnet18':
            if not np.isclose(thresholds[name]['image_threshold'], info['fixed_image_threshold_reference'], atol=1e-6, rtol=1e-6):
                raise ValueError('Reference q975 threshold reproduction failed')
            if not all(np.isclose(a['raw_score'], b['raw_score'], atol=1e-6, rtol=1e-6) for a,b in zip(calibration_rows, old_cal)):
                # CSV order is not assumed in the map/scoring check below.
                old_scores = {r['image_path']: r['raw_score'] for r in old_cal}
                if not all(np.isclose(r['raw_score'], old_scores[r['image_path']], atol=1e-6, rtol=1e-6) for r in calibration_rows):
                    raise ValueError('Reference calibration score reproduction failed')
            reproduction['normal_maps'] = compare_saved_maps(roots[name], calibration_rows, old_cal, args.baseline_dir, 'normal_calibration')
        print(f'[3/5] EVALUATE {name}: fixed45 development images, original coordinates', flush=True)
        stage('evaluate_' + name)
        rows[name], qualities[name] = evaluate(model, configs[name], thresholds[name], manifest, args.data_root, roots[name])
        if name == 'resnet18':
            reproduction['development_maps'] = compare_saved_maps(roots[name], rows[name], old_dev, args.baseline_dir, 'development')
            old_scores = {r['image_path']: r['raw_score'] for r in old_dev}
            if not all(np.isclose(r['raw_score'], old_scores[r['image_path']], atol=1e-6, rtol=1e-6) for r in rows[name]):
                raise ValueError('Reference development score reproduction failed')
        checkpoint_path = roots[name] / 'model.pt'
        torch.save({'format_version': 1, 'model_config': configs[name],
                    'model_state_dict': {k: v.detach().cpu() for k,v in model.state_dict().items()},
                    'manifest_sha256': manifest['manifest_sha256'], 'thresholds': thresholds[name]}, checkpoint_path)
        sizes[name] = {'model_parameters': sum(p.numel() for p in model.parameters()),
                       'parameter_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
                       'bank_shape': list(model.memory_bank.shape),
                       'bank_bytes': model.memory_bank.numel()*model.memory_bank.element_size(),
                       'checkpoint_bytes': checkpoint_path.stat().st_size}
        sample = manifest['splits']['development'][0]
        record = manifest['files'][sample]
        image = load_tensor(args.data_root, sample, make_transform(configs[name])).to('cuda:0')
        expected_score, expected_map = predict(model, image, (record['height'], record['width']))
        del image
        model.to('cpu')
        restored, saved_again = load_checkpoint(checkpoint_path, torch.device('cuda:0'))
        image = load_tensor(args.data_root, sample, make_transform(configs[name])).to('cuda:0')
        observed_score, observed_map = predict(restored, image, (record['height'], record['width']))
        if (not np.isclose(expected_score, observed_score, atol=1e-6, rtol=1e-6) or
                not np.allclose(expected_map, observed_map, atol=1e-6, rtol=1e-6)):
            raise ValueError('Saved/reloaded checkpoint prediction mismatch')
        checkpoint_checks[name] = {'sha256': file_sha256(checkpoint_path), 'reload_verified': True,
                                  'score_abs_difference': abs(expected_score-observed_score),
                                  'map_max_abs_difference': float(np.abs(expected_map-observed_map).max())}
        restored.to('cpu')
        del restored, saved_again, image, expected_map, observed_map
    paired = relabel_pair(paired_localization(rows['resnet18'], rows['wide_resnet50']))
    paired.update({'reference': 'resnet18', 'candidate': 'wide_resnet50'})
    write_json(output / 'paired_localization.json', paired)
    print('[4/5] BENCHMARK: four balanced AB/BA rounds; one model resident', flush=True)
    stage('benchmark')
    states = {name: state_sha256(model.state_dict()) for name, model in models.items()}
    timings, performance = benchmark(models, cfg, manifest, args.data_root, args.warmup, args.repeats)
    write_json(output / 'timings.json', timings)
    if any(state_sha256(model.state_dict()) != states[name] for name, model in models.items()):
        raise ValueError('Model state changed during inference benchmark')
    print('[5/5] VISUALIZE and VERIFY: all45 pairs, small-area zooms, source hashes', flush=True)
    stage('visualize_and_verify')
    images = visualize(output, NAMES, roots, rows, thresholds, manifest, args.data_root)
    if (any(file_sha256(safe_file(args.baseline_dir, p)) != h for p,h in identities.items()) or
            file_sha256(args.policy) != info['policy_sha256'] or
            file_sha256(args.weight_metadata) != info['weight_metadata_sha256'] or
            verify_weight_file(args.weights)['sha256'] != info['weights']['sha256'] or
            any(file_sha256(Path(__file__).with_name(p)) != h for p,h in config['source_sha256'].items())):
        raise ValueError('Frozen evidence, weights, policy or running source changed')
    config.update({'status': 'completed', 'completed_at': now()})
    write_json(output / 'config.json', config)
    left, right = (qualities[name] for name in NAMES)
    checks = {'anomaly_macro_AP_improved': right['anomaly_macro']['pixel_AP'] > left['anomaly_macro']['pixel_AP'],
              'anomaly_macro_IoU_improved': right['anomaly_macro']['pixel_IoU'] > left['anomaly_macro']['pixel_IoU'],
              'pooled_defect_pixel_recall_not_lower': right['pixel_metrics']['recall'] >= left['pixel_metrics']['recall'],
              'normal_pixel_FP_rate_not_higher': right['normal_false_positive_pixels']['rate'] <= left['normal_false_positive_pixels']['rate'],
              'no_new_image_false_negatives': right['image_counts']['FN'] <= left['image_counts']['FN'],
              'no_new_image_false_positives': right['image_counts']['FP'] <= left['image_counts']['FP']}
    report = {'status': 'completed_development_backbone_comparison_not_final_test', 'completed_at': now(),
              'config': config, 'weights_mapping': weight_mapping, 'build': build,
              'qualities': qualities, 'thresholds': thresholds, 'performance': performance, 'sizes': sizes,
              'checkpoints': checkpoint_checks, 'baseline_reproduction': reproduction,
              'quality_checks': checks, 'new_backbone_automatically_adopted': False,
              'source_evidence_unchanged': True, 'backbones_unchanged': True, 'held_out_images_opened': 0,
              'timing_boundary': 'batch1 CUDA-ready input through full reference model and original-size GPU interpolation; synchronized wall time; IO/preprocessing/transfers/CPU metrics and output writing excluded',
              'memory_boundary': 'PyTorch peak allocation with one model resident; includes weights, bank, input, outputs; not total device memory',
              'limitations': ['One category, one coreset seed, seen development split; not final-test evidence.',
                              'Only3 images below1% GT union area, nonebelow0.1%.',
                              'Both backbone architecture and ImageNet pretrained weights change; not just network depth.',
                              'Fixed1802 vectors, not equal bytes; dimension384 vs1536 and compute differ.',
                              'Independent normal-only thresholds do not enforce equal development recall or false-positive rates.',
                              'Anomalib2.2.0, custom split, concatenated1536 features and256 whole image; not full original-paper reproduction.']}
    write_json(output / 'report.json', report)
    group = '0.1_to_lt_1_percent'
    lines = ['ResNet18 / WideResNet50骨干对照已完成（整图256，开发集）',
             '176张正常建库、44张正常校准、45张开发评估；70张保留测试未使用。',
             '两组质量均在本次重新评估；ResNet18原始地图与分数复现通过。',
             '库均1802条，特征维度384→1536；不是等存储量或等算力对照。']
    for name in NAMES:
        q, perf, size = qualities[name], performance[name], sizes[name]
        small = q['area_groups'][group]
        lines.extend([f'{name}: 图像误报={q["image_counts"]["FP"]}/8，漏检={q["image_counts"]["FN"]}/37',
                      f'  异常图平均IoU={q["anomaly_macro"]["pixel_IoU"]:.4f}，F1={q["anomaly_macro"]["pixel_F1"]:.4f}，AP={q["anomaly_macro"]["pixel_AP"]:.4f}',
                      f'  缺陷像素召回={q["pixel_metrics"]["recall"]:.4f}，正常像素误报率={q["normal_false_positive_pixels"]["rate"]:.6f}',
                      f'  小面积组n={small["n"]}：IoU={small["macro"]["pixel_IoU"]:.4f}，像素召回={small["pixel_metrics"]["recall"]:.4f}',
                      f'  平均={perf["mean_ms"]:.3f}ms，P95={perf["p95_ms"]:.3f}ms，分配显存峰值={perf["max_peak_allocated_bytes"]/1024**2:.3f}MiB',
                      f'  库={size["bank_bytes"]/1024**2:.3f}MiB，保存模型={size["checkpoint_bytes"]/1024**2:.3f}MiB'])
    lines += [f'质量检查：{checks}', '没有自动替换当前配置。IoU、AP、召回、误报与成本一起判断；小面积仅3张，不能宣称泛化提升。']
    (output / '结果摘要.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_html(output, '骨干网络对照：ResNet18 / WideResNet50', lines, images)
    write_json(output / 'status.json', {'status': 'completed', 'completed_at': now()})
    print('\n'.join(lines), flush=True)
    print(f'BACKBONE_COMPARISON_COMPLETED | {output}', flush=True)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=root / '04实验结果/01模型与检索实验/baseline_20261016_204723_088238')
    parser.add_argument('--data-root', type=Path, default=root / '03原始数据/解压数据/MVTecAD')
    parser.add_argument('--manifest', type=Path, default=root / '02代码与配置/configs/splits/metal_nut-v1.json')
    parser.add_argument('--policy', type=Path, default=root / '02代码与配置/configs/decisions/metal_nut-q975-provisional.json')
    parser.add_argument('--weights', type=Path, default=root / '.cache/models/torch/hub/checkpoints' / WEIGHT_NAME)
    parser.add_argument('--weight-metadata', type=Path, default=root / '02代码与配置/configs/pretrained/wide-resnet50-tv-in1k.json')
    parser.add_argument('--output-root', type=Path, default=root / '04实验结果/09骨干网络对照')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2 or args.repeats % 2:
        parser.error('Positive warmup and even >=2 timing rounds required')
    for key in ('baseline_dir', 'data_root', 'manifest', 'policy', 'weights', 'weight_metadata', 'output_root'):
        setattr(args, key, getattr(args, key).resolve())
    if not args.output_root.is_relative_to(root / '04实验结果') or any(
            args.output_root == p or args.output_root.is_relative_to(p) for p in (args.baseline_dir, args.data_root)):
        parser.error('Use a new result directory under project04, outside baseline and raw data')
    inputs, info = preflight(args)
    if args.check_only:
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        print('BACKBONE_PREFLIGHT_PASSED | no fitting or forward', flush=True)
        return
    output = args.output_root / ('backbone_comparison_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f'))
    print(f'OUTPUT: {output}', flush=True)
    try:
        run(args, inputs, info, output)
    except Exception as error:
        if output.exists():
            write_json(output / 'status.json', {'status': 'failed_not_completed', 'at': now(), 'error_type': type(error).__name__, 'error': str(error)})
            (output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
