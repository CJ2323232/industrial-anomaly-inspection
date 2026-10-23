"""Manual fixed-bank-budget 256/512 PatchCore development comparison.

Uses the existing 256 checkpoint and its frozen backbone to build a 512 bank.
Normal-only calibration is separate for each resolution. No held-out inference.
"""
import argparse
import copy
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np

if __package__:
    from .analyze_thresholds import checked_inputs
    from .baseline_utils import (area_group, binary_counts, binary_metrics, file_sha256,
                                 load_manifest, normal_threshold, safe_file, verify_consumed_files)
    from .resolution_metrics import evaluate_records, paired_localization
    from .run_trusted_baseline import (load_checkpoint, load_mask, load_tensor, make_transform,
                                      predict, state_sha256, write_json)
else:
    from analyze_thresholds import checked_inputs
    from baseline_utils import (area_group, binary_counts, binary_metrics, file_sha256,
                                load_manifest, normal_threshold, safe_file, verify_consumed_files)
    from resolution_metrics import evaluate_records, paired_localization
    from run_trusted_baseline import (load_checkpoint, load_mask, load_tensor, make_transform,
                                     predict, state_sha256, write_json)

RESOLUTIONS = (256, 512)
IMAGE_Q, PIXEL_Q = .975, .995


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inspect_inputs(args):
    import torch
    identities, report, calibration_rows, development_rows = checked_inputs(args.baseline_dir)
    manifest = load_manifest(args.manifest)
    if manifest['manifest_sha256'] != report['manifest_sha256']:
        raise ValueError('Requested split does not match the frozen baseline')
    config = report['config']
    if (config['image_size'] != [256, 256] or config['layers'] != ['layer2', 'layer3']
            or config['num_neighbors'] != 9 or report['memory_bank_shape'] != [1802, 384]
            or report['candidate_feature_count'] != 180224 or config['seed'] != 0):
        raise ValueError('This comparison requires the verified 256/1802x384/seed0 baseline')
    if version('anomalib') != '2.2.0' or version('anomalib') != report['environment']['anomalib']:
        raise ValueError('Use the baseline anomalib 2.2.0 environment')
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for this resource comparison')
    verified = verify_consumed_files(manifest, args.data_root)
    small_counts = {name: 0 for name in ('lt_0.1_percent', '0.1_to_lt_1_percent', 'ge_1_percent')}
    for row in development_rows:
        if row['label']:
            small_counts[area_group(float(row['area_ratio']))] += 1
    info = {'status': 'resolution_preflight_passed_not_fitted', 'checked_at': now(), **verified,
            'manifest_sha256': manifest['manifest_sha256'], 'checkpoint_sha256': identities['model.pt'],
            'resolutions': list(RESOLUTIONS), 'bank_vectors_each': 1802, 'feature_dim': 384,
            'candidate_counts': {'256': 180224, '512': 720896},
            'historical_development_area_counts': small_counts,
            'image_quantile': IMAGE_Q, 'pixel_quantile': PIXEL_Q,
            'gpu': torch.cuda.get_device_name(0), 'anomalib': version('anomalib')}
    return identities, report, manifest, calibration_rows, development_rows, info


def select_bank(embedding, count, seed=0):
    """Use reference coreset selection with an explicit cardinality, not ratio rounding."""
    import torch
    from anomalib.models.components import KCenterGreedy
    if (embedding.ndim != 2 or not 1 <= count <= len(embedding)
            or not torch.isfinite(embedding).all()):
        raise ValueError('Invalid coreset candidates or cardinality')
    seed_all(seed)
    sampler = KCenterGreedy(embedding=embedding, sampling_ratio=count / len(embedding))
    sampler.coreset_size = count
    indices = sampler.select_coreset_idxs()
    if len(indices) != count or len(set(indices)) != count:
        raise ValueError('Reference sampler did not return the requested distinct bank rows')
    bank = embedding[indices].clone()
    metadata = {'algorithm': 'anomalib-2.2.0-KCenterGreedy', 'vectors': count,
                'effective_sampling_ratio': count / len(embedding), 'seed_reset_before_sampling': seed,
                'projection_dim': int(sampler.model.n_components),
                'projection_sha256': state_sha256({'projection': sampler.model.sparse_random_matrix}),
                'bank_sha256': state_sha256({'bank': bank})}
    return bank, indices, metadata


def build_512(model, config, manifest, data_root, output, bank_count):
    import torch
    model.to('cuda:0').requires_grad_(False)
    model.memory_bank = torch.empty(0, device='cuda:0')
    model.train()
    model.feature_extractor.eval()
    transform = make_transform(config)
    paths = manifest['splits']['fit']
    patches = (config['image_size'][0] // 8) ** 2
    candidates = torch.empty((len(paths) * patches, 384), dtype=torch.float32, device='cpu')
    offset = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before_bytes = torch.cuda.memory_allocated()
    start = time.perf_counter()
    with torch.no_grad():
        for index in range(0, len(paths), config['fit_batch_size']):
            batch_paths = paths[index:index + config['fit_batch_size']]
            batch = torch.stack([load_tensor(data_root, p, transform) for p in batch_paths]).to('cuda:0')
            embedding = model(batch)
            if tuple(embedding.shape) != (len(batch_paths) * patches, 384):
                raise ValueError('Unexpected candidate feature grid')
            candidates[offset:offset + len(embedding)].copy_(embedding.detach().cpu())
            offset += len(embedding)
            model.embedding_store.clear()  # Avoid retaining the full pool on GPU.
            del embedding, batch
            if (index + len(batch_paths)) % 44 == 0 or index + len(batch_paths) == len(paths):
                print(f'  normal features {index + len(batch_paths)}/{len(paths)}', flush=True)
        if offset != len(candidates):
            raise ValueError('Candidate feature count differs from the plan')
        device_candidates = candidates.to('cuda:0')
        del candidates
        bank, indices, metadata = select_bank(device_candidates, bank_count, seed=0)
        model.memory_bank = bank
        del device_candidates
    model.eval()
    torch.cuda.synchronize()
    metadata.update({'seconds': time.perf_counter() - start,
                     'candidate_feature_count': offset, 'bank_shape': list(bank.shape),
                     'bank_bytes': bank.numel() * bank.element_size(),
                     'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                     'allocated_before_bytes': int(before_bytes),
                     'peak_boundary': '512 feature extraction and coreset GPU allocation, including resident model; CPU candidate staging excluded',
                     'duration_boundary': '512 image reads/preprocessing, feature extraction, staging/transfers and coreset selection; excludes calibration/evaluation'})
    write_json(output / 'coreset_indices.json', {'indices': indices, 'fit_paths_in_order': paths,
                                               'patches_per_image': patches,
                                               'flattening': 'image order, row-major spatial grid'})
    metadata['coreset_indices_file_sha256'] = file_sha256(output / 'coreset_indices.json')
    write_json(output / 'build_report.json', metadata)
    return metadata


def save_map(output, split, relative, array):
    destination = output / 'maps' / split / Path(relative).with_suffix('.npz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, anomaly_map=array)
    return destination.relative_to(output).as_posix()


def calibrate(model, config, manifest, data_root, output):
    transform = make_transform(config)
    paths = manifest['splits']['normal_calibration']
    pixel_values = np.empty(sum(manifest['files'][p]['height'] * manifest['files'][p]['width'] for p in paths), dtype=np.float32)
    offset, rows = 0, []
    for index, relative in enumerate(paths):
        record = manifest['files'][relative]
        score, score_map = predict(model, load_tensor(data_root, relative, transform).to('cuda:0'),
                                   (record['height'], record['width']))
        pixel_values[offset:offset + score_map.size] = score_map.reshape(-1)
        offset += score_map.size
        rows.append({'image_path': relative, 'label': 0, 'raw_score': score,
                     'image_sha256': record['image_sha256'], 'map_path': save_map(output, 'normal_calibration', relative, score_map)})
        if (index + 1) % 11 == 0:
            print(f'  calibration {config["image_size"][0]}: {index+1}/{len(paths)}', flush=True)
    thresholds = {'image_quantile': IMAGE_Q, 'pixel_quantile': PIXEL_Q, 'quantile_method': 'linear',
                  'image_threshold': normal_threshold([r['raw_score'] for r in rows], [0] * len(rows), IMAGE_Q),
                  'pixel_threshold': float(np.quantile(pixel_values, PIXEL_Q, method='linear')),
                  'decision_rule': 'raw_score > threshold', 'normal_images': len(paths),
                  'normal_pixels': int(pixel_values.size), 'frozen_at': now()}
    thresholds['normal_image_exceedances'] = sum(r['raw_score'] > thresholds['image_threshold'] for r in rows)
    thresholds['normal_pixel_exceedance_fraction'] = float(np.count_nonzero(pixel_values > thresholds['pixel_threshold']) / pixel_values.size)
    write_json(output / 'calibration.json', thresholds)
    write_json(output / 'calibration_predictions.json', rows)
    return thresholds, rows


def evaluate(model, config, thresholds, manifest, data_root, output):
    transform = make_transform(config)
    paths = manifest['splits']['development']
    count = sum(manifest['files'][p]['height'] * manifest['files'][p]['width'] for p in paths)
    labels_all, scores_all = np.empty(count, dtype=np.uint8), np.empty(count, dtype=np.float32)
    offset, rows = 0, []
    for index, relative in enumerate(paths):
        record = manifest['files'][relative]
        mask = load_mask(data_root, record)
        score, score_map = predict(model, load_tensor(data_root, relative, transform).to('cuda:0'),
                                   (record['height'], record['width']))
        pixel_metrics = binary_metrics(mask.reshape(-1), score_map.reshape(-1), thresholds['pixel_threshold'])
        labels_all[offset:offset + mask.size], scores_all[offset:offset + mask.size] = mask.reshape(-1), score_map.reshape(-1)
        offset += mask.size
        rows.append({'image_path': relative, 'label': record['label'], 'defect_type': record['defect_type'],
                     'image_sha256': record['image_sha256'], 'mask_sha256': record['mask_sha256'],
                     'area_ratio': float(mask.mean()), 'raw_score': score,
                     'pred_label': int(score > thresholds['image_threshold']),
                     'pixel_counts': {key: pixel_metrics[key] for key in ('TP', 'FP', 'FN', 'TN')},
                     'pixel_AP': pixel_metrics['AP'], 'pixel_AUROC': pixel_metrics['AUROC'],
                     'map_path': save_map(output, 'development', relative, score_map)})
        if (index + 1) % 10 == 0 or index + 1 == len(paths):
            print(f'  development {config["image_size"][0]}: {index+1}/{len(paths)}', flush=True)
    summary = evaluate_records(rows)
    print(f'  exact pooled pixel AUROC/AP at {config["image_size"][0]} (CPU)', flush=True)
    summary['pooled_pixel_metrics'] = binary_metrics(labels_all, scores_all, thresholds['pixel_threshold'])
    summary['image_continuous_metrics'] = binary_metrics([r['label'] for r in rows], [r['raw_score'] for r in rows], thresholds['image_threshold'])
    write_json(output / 'development_predictions.json', rows)
    write_json(output / 'quality_report.json', summary)
    return rows, summary


def benchmark(models, configs, manifest, data_root, warmup=10, repeats=3):
    """Alternate whole-round order; only one model on GPU, swaps outside timing."""
    import torch
    from torch.nn import functional as F
    paths = manifest['splits']['development']
    cpu_images = {size: [load_tensor(data_root, p, make_transform(configs[size])) for p in paths] for size in RESOLUTIONS}
    def forward(model, image, shape):
        result = model(image)
        return result.pred_score, F.interpolate(result.anomaly_map, size=shape, mode='bilinear', align_corners=False)
    records = []
    with torch.inference_mode():
        for repeat in range(repeats):
            order = RESOLUTIONS if repeat % 2 == 0 else RESOLUTIONS[::-1]
            for size in order:
                for other, model in models.items():
                    model.to('cpu')
                model = models[size].to('cuda:0').eval()
                for n in range(warmup):
                    index = n % len(paths)
                    record = manifest['files'][paths[index]]
                    image = cpu_images[size][index].unsqueeze(0).to('cuda:0')
                    result = forward(model, image, (record['height'], record['width']))
                    torch.cuda.synchronize()
                    del image, result
                for relative, cpu_image in zip(paths, cpu_images[size]):
                    record = manifest['files'][relative]
                    image = cpu_image.unsqueeze(0).to('cuda:0')
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    before = torch.cuda.memory_allocated()
                    start = time.perf_counter()
                    result = forward(model, image, (record['height'], record['width']))
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    peak = torch.cuda.max_memory_allocated()
                    records.append({'resolution': size, 'repeat': repeat, 'image_path': relative,
                                    'milliseconds': elapsed, 'peak_allocated_bytes': int(peak),
                                    'allocated_before_bytes': int(before), 'incremental_peak_bytes': int(peak-before)})
                    del result, image
                model.to('cpu')
                print(f'  benchmark round {repeat+1}/{repeats}, resolution {size}', flush=True)
    summary = {}
    for size in RESOLUTIONS:
        selected = [r for r in records if r['resolution'] == size]
        times = [r['milliseconds'] for r in selected]
        summary[str(size)] = {'samples': len(times), 'mean_ms': float(np.mean(times)),
                              'p95_ms': float(np.quantile(times, .95, method='linear')),
                              'max_peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in selected),
                              'max_incremental_peak_bytes': max(r['incremental_peak_bytes'] for r in selected),
                              'round_mean_ms': [float(np.mean([r['milliseconds'] for r in selected if r['repeat'] == k])) for k in range(repeats)]}
    return records, summary


def visualize_pairs(output, rows, manifest, data_root, thresholds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    paired = {r['image_path']: r for r in rows}
    # Fixed small-area group plus five lowest baseline IoU cases; save all numbers separately.
    anomalies = [r for r in rows if r['label']]
    selected = {r['image_path'] for r in anomalies if r['area_ratio'] < .01}
    selected.update(r['image_path'] for r in sorted(anomalies, key=lambda r: r['pixel_IoU_256'])[:5])
    index = []
    for relative in sorted(selected):
        row = paired[relative]
        mask = load_mask(data_root, manifest['files'][relative])
        with Image.open(safe_file(data_root, relative)) as im:
            original = np.asarray(im)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original')
        axes[1, 0].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Ground truth (original coordinates)')
        for y, size in enumerate(RESOLUTIONS):
            with np.load(output / str(size) / 'maps/development' / Path(relative).with_suffix('.npz')) as maps:
                score_map = maps['anomaly_map']
            tau = thresholds[size]['pixel_threshold']
            axes[y, 1].imshow(original)
            axes[y, 1].imshow(score_map / max(tau, 1e-12), cmap='turbo', vmin=0, vmax=2, alpha=.55)
            axes[y, 1].set_title(f'{size}: map / pixel threshold (display only)')
            axes[y, 2].imshow(score_map > tau, cmap='gray', vmin=0, vmax=1)
            axes[y, 2].contour(mask, levels=[.5], colors=['red'], linewidths=.6)
            axes[y, 2].set_title(f'{size}: mask, IoU={row[f"pixel_IoU_{size}"]:.4f}; red=GT')
        for axis in axes.flat:
            axis.axis('off')
        fig.suptitle(f'{relative} | GT union area={row["area_ratio"]*100:.3f}%')
        fig.tight_layout()
        destination = output / '可视化' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        index.append({'image_path': relative, 'visualization': destination.relative_to(output).as_posix(),
                      'selection': 'GT union area <1% or five lowest baseline IoU'})
    write_json(output / 'visualization_index.json', index)


def run_experiment(args, inputs, output):
    import torch
    identities, original_report, manifest, old_calibration, old_development, preflight = inputs
    output.mkdir(parents=True, exist_ok=False)
    def stage(name):
        write_json(output / 'status.json', {'status': 'running', 'stage': name, 'at': now()})
    stage('initializing')
    write_json(output / 'preflight.json', preflight)
    write_json(output / 'split_manifest.json', manifest)
    config = {'status': 'started_not_completed', 'started_at': now(), 'resolutions': [256,512],
              'comparison': 'default PatchCore pipeline resolution at fixed 1802-vector bank budget',
              'image_quantile': IMAGE_Q, 'pixel_quantile': PIXEL_Q, 'warmup_each_round_each_resolution': args.warmup,
              'repeats': args.repeats, 'fit_batch_size_512': 4, 'seed_512_coreset': 0,
              'baseline_sources_sha256': identities, 'manifest_sha256': manifest['manifest_sha256'],
              'source_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in
                                ('compare_resolution.py', 'resolution_metrics.py', 'run_trusted_baseline.py', 'baseline_utils.py', 'analyze_thresholds.py')},
              'environment': {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'anomalib': version('anomalib'),
                              'numpy': version('numpy'), 'sklearn': version('scikit-learn'), 'gpu': torch.cuda.get_device_name(0)}}
    write_json(output / 'config.json', config)
    seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    models, configs, thresholds, results = {}, {}, {}, {}
    for size in RESOLUTIONS:
        models[size], saved = load_checkpoint(args.baseline_dir / 'model.pt', torch.device('cpu'))
        models[size].requires_grad_(False)
        if saved['model_config'] != original_report['config'] or saved['manifest_sha256'] != manifest['manifest_sha256']:
            raise ValueError('Loaded checkpoint configuration identity mismatch')
        if state_sha256(models[size].feature_extractor.state_dict()) != original_report['frozen_backbone_state_sha256']:
            raise ValueError('Loaded backbone identity mismatch')
        configs[size] = copy.deepcopy(saved['model_config'])
        configs[size].update({'image_size': [size,size], 'image_quantile': IMAGE_Q})
        (output / str(size)).mkdir()
        write_json(output / str(size) / 'config.json', configs[size])
    original_bank_sha = state_sha256({'bank': models[256].memory_bank})
    print('[1/4] BUILD512: 176 normal images; exactly 1802 bank vectors', flush=True)
    stage('build_512')
    build = build_512(models[512], configs[512], manifest, args.data_root, output / '512', 1802)
    configs[512].update({'coreset_ratio': build['effective_sampling_ratio'], 'candidate_feature_count': build['candidate_feature_count'],
                         'bank_vector_budget': 1802, 'coreset_seed_reset_before_sampling': 0})
    write_json(output / '512/config.json', configs[512])
    models[512].to('cpu')
    bank_hashes = {size: state_sha256({'bank': model.memory_bank}) for size,model in models.items()}
    print('[2/4] CALIBRATE: separate normal-only q=.975 / pixel q=.995', flush=True)
    stage('calibrate')
    reproduction = {}
    for size in RESOLUTIONS:
        model = models[size].to('cuda:0').eval()
        thresholds[size], cal_rows = calibrate(model, configs[size], manifest, args.data_root, output / str(size))
        if size == 256:
            old = {r['image_path']: r['raw_score'] for r in old_calibration}
            reproduction['calibration_scores_close'] = all(np.isclose(r['raw_score'], old[r['image_path']], atol=1e-4, rtol=1e-5) for r in cal_rows)
            reproduction['pixel_threshold_close'] = bool(np.isclose(thresholds[size]['pixel_threshold'], original_report['thresholds']['pixel_threshold'], atol=1e-4, rtol=1e-5))
            reproduction['candidate_image_threshold_close'] = bool(np.isclose(thresholds[size]['image_threshold'], normal_threshold(list(old.values()), [0]*len(old), IMAGE_Q), atol=1e-4, rtol=1e-5))
            if not all(reproduction.values()):
                raise ValueError('Existing 256 calibration could not be reproduced')
        model.to('cpu')
    checkpoint = {'format_version': 1, 'model_config': configs[512],
                  'model_state_dict': models[512].state_dict(), 'manifest_sha256': manifest['manifest_sha256'],
                  'thresholds': thresholds[512]}
    torch.save(checkpoint, output / '512/model.pt')
    del checkpoint
    write_json(output / '256/model_reference.json', {'checkpoint_sha256': identities['model.pt'], 'existing_checkpoint_reused': True})
    print('[3/4] EVALUATE: same 45 development images, original coordinates', flush=True)
    stage('evaluate')
    qualities = {}
    for size in RESOLUTIONS:
        model = models[size].to('cuda:0').eval()
        results[size], qualities[size] = evaluate(model, configs[size], thresholds[size], manifest, args.data_root, output / str(size))
        model.to('cpu')
        if size == 256:
            old = {r['image_path']: r['raw_score'] for r in old_development}
            reproduction['development_scores_close'] = all(np.isclose(r['raw_score'], old[r['image_path']], atol=1e-4, rtol=1e-5) for r in results[size])
            if not reproduction['development_scores_close']:
                raise ValueError('Existing 256 development scores could not be reproduced')
    paired = paired_localization(results[256], results[512])
    write_json(output / 'paired_localization.json', paired)
    print('[4/4] BENCHMARK: one GPU-resident model; round order AB/BA/AB', flush=True)
    stage('benchmark')
    timing_rows, performance = benchmark(models, configs, manifest, args.data_root, args.warmup, args.repeats)
    write_json(output / 'timings.json', timing_rows)
    stage('visualization_and_integrity')
    visualize_pairs(output, paired['per_image'], manifest, args.data_root, thresholds)
    for size, model in models.items():
        if (state_sha256(model.feature_extractor.state_dict()) != original_report['frozen_backbone_state_sha256']
                or state_sha256({'bank': model.memory_bank}) != bank_hashes[size]):
            raise ValueError('Model changed during evaluation')
    if original_bank_sha != bank_hashes[256] or any(file_sha256(args.baseline_dir/name) != sha for name,sha in identities.items()):
        raise ValueError('Existing baseline changed')
    loaded, _ = load_checkpoint(output / '512/model.pt', torch.device('cpu'))
    if state_sha256(loaded.state_dict()) != state_sha256(models[512].state_dict()):
        raise ValueError('512 checkpoint reload state mismatch')
    del loaded
    config['status'] = 'completed'
    config['completed_at'] = now()
    write_json(output / 'config.json', config)
    report = {'status': 'completed_development_resolution_comparison_not_final_test',
              'completed_at': now(), 'config': config, 'baseline_reproduction': reproduction,
              'qualities': {str(s): qualities[s] for s in RESOLUTIONS},
              'thresholds': {str(s): thresholds[s] for s in RESOLUTIONS},
              'performance': performance, 'build_512': build,
              'build_256_comparable_measurement': None, 'checkpoint_512_sha256': file_sha256(output / '512/model.pt'),
              'checkpoint_512_reload_state_verified': True,
              'paired_localization_file': 'paired_localization.json',
              'development_images': len(manifest['splits']['development']), 'held_out_images_opened': 0,
              'automatically_selected_resolution': None,
              'timing_boundary': 'batch1 CUDA-ready input through reference dense model and original-size GPU interpolation; synchronized wall time; excludes decoding/preprocessing/H2D/D2H/metrics/writing/model swapping',
              'memory_boundary': 'Measured PyTorch peak allocated bytes with one model resident; includes model/input/output; incremental subtracts pre-call allocation; not total VRAM or CPU RAM',
              'limitations': ['Only three development images have GT union area 0.1%-<1%; none below0.1%.',
                              'Image union area is not individual connected defect size.',
                              'Fixed1802bank budget changes effective coreset ratio from~1% to~0.25%.',
                              'Reference random projection dimension and RNG stream differ with candidates; not a pure optical-resolution ablation.',
                              'Default sigma4 remains in input pixels, so original-space smoothing and receptive fields change.',
                              'The q.975 rule was already selected using development feedback.',
                              'One category and one coreset seed; seen development data, not a final test.',
                              'No assumed quality improvement; inspect paired results and resource tradeoffs.']}
    write_json(output / 'report.json', report)
    small = '0.1_to_lt_1_percent'
    lines = ['256 / 512 分辨率对照已完成（固定1802特征库预算，开发集实验）',
             '两组阈值分别由44张正常校准图确定；保留测试70张未使用。']
    def metric_text(values):
        return ', '.join(f'{name}={values["pixel_"+name]:.4f}' if values['pixel_'+name] is not None
                         else f'{name}=N/A' for name in ('IoU', 'F1', 'AP'))
    for size in RESOLUTIONS:
        q, perf = qualities[size], performance[str(size)]
        group = q['area_groups'][small]
        lines.append(f'{size}: 图像误报={q["image_counts"]["FP"]}/{q["normal_images"]}，漏检={q["image_counts"]["FN"]}/{q["anomalous_images"]}')
        lines.append(f'  全部异常图逐图平均：{metric_text(q["anomaly_macro"])}')
        lines.append(f'  小面积组 n={group["n"]}，逐图平均：{metric_text(group["macro"])}')
        lines.append(f'  平均={perf["mean_ms"]:.3f}ms, P95={perf["p95_ms"]:.3f}ms, 分配显存峰值={perf["max_peak_allocated_bytes"]/1024**2:.3f}MiB')
    lines += ['小面积组仅3张，没有面积小于0.1%的图；不能据此宣称对极小缺陷普遍有效。',
              '查看 paired_localization.json 和 可视化/ 对比逐图变化；没有自动替换当前方案。',
              '耗时不含解码、预处理和数据传输；显存是PyTorch分配峰值。']
    (output / '结果摘要.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(output / 'status.json', {'status': 'completed', 'completed_at': now()})
    print('\n'.join(lines), flush=True)
    print(f'RESOLUTION_COMPARISON_COMPLETED | {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if min(args.warmup, args.repeats) < 1:
        parser.error('Warmup and repeats must be positive')
    args.baseline_dir, args.data_root, args.manifest = args.baseline_dir.resolve(), args.data_root.resolve(), args.manifest.resolve()
    inputs = inspect_inputs(args)
    if args.check_only:
        print(json.dumps(inputs[-1], ensure_ascii=False, indent=2), flush=True)
        print('RESOLUTION_PREFLIGHT_PASSED | no fitting, model forward or held-out evaluation', flush=True)
        return
    if args.output_root is None:
        parser.error('--output-root is required')
    output_root = args.output_root.resolve()
    if any(output_root == p or output_root.is_relative_to(p) for p in (args.baseline_dir,args.data_root)):
        parser.error('Output must be outside the original baseline and raw data')
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = output_root / f'resolution_comparison_{stamp}'
    print(f'OUTPUT: {output}', flush=True)
    try:
        run_experiment(args, inputs, output)
    except Exception as error:
        if output.exists():
            write_json(output / 'status.json', {'status': 'failed_not_completed', 'at': now(),
                                              'error_type': type(error).__name__, 'error': str(error)})
            (output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
