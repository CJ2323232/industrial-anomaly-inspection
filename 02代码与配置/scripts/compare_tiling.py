"""One prespecified overlap-tiling experiment; development only, never held-out."""
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
    from .baseline_utils import file_sha256, safe_file
    from .compare_resolution import inspect_inputs, select_bank, seed_all, calibrate, evaluate
    from .resolution_metrics import paired_localization
    from .run_trusted_baseline import load_checkpoint, load_tensor, load_mask, make_transform, state_sha256, write_json
    from .tiling_backend import OverlapTiler, windows, sampled_patch_indices
else:
    from baseline_utils import file_sha256, safe_file
    from compare_resolution import inspect_inputs, select_bank, seed_all, calibrate, evaluate
    from resolution_metrics import paired_localization
    from run_trusted_baseline import load_checkpoint, load_tensor, load_mask, make_transform, state_sha256, write_json
    from tiling_backend import OverlapTiler, windows, sampled_patch_indices

METHODS = ('whole256', 'whole512', 'tiled256')
PLAN = {'tile_size': 256, 'stride': 224, 'image_size': [700, 700],
        'axis_starts': [0, 224, 444], 'tiles_per_image': 9,
        'tile_batch_size': 1, 'resize_before_tiling': False,
        'fusion': 'separable triangular weights, each axis clamped >=0.05; normalized overlap sum',
        'image_score': 'maximum of nine reference weighted PatchCore tile scores',
        'sigma_each_tile': 4, 'fit_candidate_budget_per_image': 1024,
        'prefilter': 'uniform spatial sampling without replacement, balanced across tiles, seed=image index',
        'fit_candidate_count': 180224, 'bank_vectors': 1802, 'feature_dim': 384,
        'coreset_seed': 0, 'image_quantile': .975, 'pixel_quantile': .995,
        'quantile_method': 'linear', 'decision_rule': 'strict greater-than',
        'automatic_adoption': False, 'held_out_images_opened': 0}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def preflight(args):
    inputs = inspect_inputs(args)
    identities, baseline, manifest, old_cal, old_dev, info = inputs
    resolution = load_json(args.resolution_dir / 'report.json')
    source_manifest = load_json(args.resolution_dir / 'split_manifest.json')
    if (resolution['status'] != 'completed_development_resolution_comparison_not_final_test'
            or source_manifest['manifest_sha256'] != manifest['manifest_sha256']):
        raise ValueError('Require the completed paired resolution run and identical split')
    if any((manifest['files'][p]['height'], manifest['files'][p]['width']) != (700, 700)
           for split in ('fit', 'normal_calibration', 'development') for p in manifest['splits'][split]):
        raise ValueError('This prespecified experiment requires native700x700 images')
    sources = {}
    for size in ('256', '512'):
        for name in ('development_predictions.json', 'quality_report.json', 'calibration.json'):
            path = args.resolution_dir / size / name
            sources[path.relative_to(args.resolution_dir).as_posix()] = file_sha256(path)
        rows = load_json(args.resolution_dir / size / 'development_predictions.json')
        if {r['image_path'] for r in rows} != set(manifest['splits']['development']):
            raise ValueError('Stored whole-image rows disagree with development split')
        for row in rows:
            record = manifest['files'][row['image_path']]
            if row['image_sha256'] != record['image_sha256'] or row['mask_sha256'] != record['mask_sha256']:
                raise ValueError('Stored evaluation image/mask identity mismatch')
            sources[f'{size}/{row["map_path"]}'] = file_sha256(safe_file(args.resolution_dir / size, row['map_path']))
    for name in ('512/model.pt', 'report.json', 'split_manifest.json'):
        sources[name] = file_sha256(args.resolution_dir / name)
    if file_sha256(args.resolution_dir / '512/model.pt') != resolution['checkpoint_512_sha256']:
        raise ValueError('512 checkpoint identity changed')
    info.update({'status': 'tiling_preflight_passed_no_forward', 'plan': PLAN,
                 'resolution_source_sha256': sources,
                 'real_run_authorized_by_user': '直接帮我做一个重叠分块方案看看效果'})
    return inputs, sources, info


def native_config(config):
    result = copy.deepcopy(config)
    result.update({'image_size': [700, 700], 'image_quantile': .975, 'pixel_quantile': .995})
    return result


def build_bank(model, config, manifest, data_root, output):
    model.to('cuda:0').requires_grad_(False)
    model.memory_bank = torch.empty(0, device='cuda:0')
    model.train()
    model.feature_extractor.eval()
    transform = make_transform(native_config(config))
    candidates = torch.empty((PLAN['fit_candidate_count'], 384), dtype=torch.float32)
    rectangles = windows(700, 700)
    sampling_records, offset = [], 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for index, relative in enumerate(manifest['splits']['fit']):
            image = load_tensor(data_root, relative, transform).unsqueeze(0).to('cuda:0')
            selected = sampled_patch_indices(9, seed=index)
            for tile_index, (y, x, yy, xx) in enumerate(rectangles):
                embedding = model(image[:, :, y:yy, x:xx].contiguous())
                if tuple(embedding.shape) != (1024, 384):
                    raise ValueError('Unexpected tile feature layout')
                indices = selected[tile_index]
                candidates[offset:offset + len(indices)].copy_(embedding[indices].cpu())
                offset += len(indices)
                model.embedding_store.clear()
                del embedding
            sampling_records.append({'image_path': relative, 'patch_indices_per_tile': selected})
            del image
            if (index + 1) % 22 == 0:
                print(f'  normal tiled features {index+1}/176', flush=True)
        if offset != len(candidates):
            raise ValueError('Candidate count mismatch')
        pool = candidates.to('cuda:0')
        del candidates
        bank, indices, metadata = select_bank(pool, 1802, seed=0)
        model.memory_bank = bank
        del pool
    model.eval()
    torch.cuda.synchronize()
    metadata.update({'seconds': time.perf_counter() - started,
                     'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                     'all_tile_features_before_prefilter': 176 * 9 * 1024,
                     'candidate_feature_count': offset, 'bank_shape': list(bank.shape),
                     'bank_bytes': bank.numel() * bank.element_size(),
                     'peak_boundary': 'PyTorch allocated GPU memory during extraction and coreset; excludes CPU staging',
                     'sampling_is_extra_method_change': True})
    write_json(output / 'candidate_sampling.json', {'tile_windows_yx_yx': rectangles, 'images': sampling_records})
    write_json(output / 'coreset_indices.json', {'indices': indices, 'indexing': 'prefiltered image/tile/patch order'})
    metadata['candidate_sampling_sha256'] = file_sha256(output / 'candidate_sampling.json')
    metadata['coreset_indices_sha256'] = file_sha256(output / 'coreset_indices.json')
    write_json(output / 'build_report.json', metadata)
    return metadata


def forward_original(runner, image, shape):
    output = runner(image)
    return output.pred_score, F.interpolate(output.anomaly_map, size=shape, mode='bilinear', align_corners=False)


def benchmark(models, configs, manifest, data_root, warmup, repeats):
    paths = manifest['splits']['development']
    records = []
    with torch.inference_mode():
        for repeat in range(repeats):
            order = METHODS[repeat % 3:] + METHODS[:repeat % 3]
            for name in order:
                for model in models.values():
                    model.to('cpu')
                torch.cuda.empty_cache()
                model = models[name].to('cuda:0').eval()
                runner = OverlapTiler(model) if name == 'tiled256' else model
                transform = make_transform(configs[name])
                for index in range(warmup):
                    image = load_tensor(data_root, paths[index % len(paths)], transform).unsqueeze(0).to('cuda:0')
                    output = forward_original(runner, image, (700, 700))
                    torch.cuda.synchronize()
                    del image, output
                for relative in paths:
                    image = load_tensor(data_root, relative, transform).unsqueeze(0).to('cuda:0')
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    before = torch.cuda.memory_allocated()
                    started = time.perf_counter()
                    output = forward_original(runner, image, (700, 700))
                    torch.cuda.synchronize()
                    records.append({'method': name, 'repeat': repeat, 'image_path': relative,
                                    'milliseconds': (time.perf_counter() - started) * 1000,
                                    'allocated_before_bytes': int(before),
                                    'peak_allocated_bytes': int(torch.cuda.max_memory_allocated())})
                    del image, output
                del runner
                model.to('cpu')
                print(f'  benchmark round {repeat+1}/{repeats}: {name}', flush=True)
    summary = {}
    for name in METHODS:
        rows = [r for r in records if r['method'] == name]
        summary[name] = {'samples': len(rows), 'mean_ms': float(np.mean([r['milliseconds'] for r in rows])),
                         'p95_ms': float(np.quantile([r['milliseconds'] for r in rows], .95)),
                         'max_peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in rows),
                         'round_mean_ms': [float(np.mean([r['milliseconds'] for r in rows if r['repeat'] == n]))
                                           for n in range(repeats)]}
    return records, summary


def visualize(output, roots, rows, thresholds, manifest, data_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    by_method = {name: {r['image_path']: r for r in records} for name, records in rows.items()}
    # Show every anomalous development image, not only favorable examples.
    index = []
    for relative in manifest['splits']['development']:
        record = manifest['files'][relative]
        mask = load_mask(data_root, record).astype(bool)
        with Image.open(safe_file(data_root, relative)) as im:
            original = np.asarray(im)
        maps = {}
        for name in METHODS:
            row = by_method[name][relative]
            with np.load(safe_file(roots[name], row['map_path'])) as z:
                maps[name] = z['anomaly_map']
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        for n, name in enumerate(METHODS):
            row = by_method[name][relative]
            tau = thresholds[name]['pixel_threshold']
            predicted = maps[name] > tau
            axes[n, 0].imshow(original)
            if mask.any():
                axes[n, 0].contour(mask, levels=[.5], colors=['yellow'], linewidths=.6)
            axes[n, 0].set_title(f'{name}: original / GT yellow')
            axes[n, 1].imshow(maps[name] / tau, cmap='turbo', vmin=0, vmax=2)
            axes[n, 1].set_title('map / own threshold (display only)')
            axes[n, 2].imshow(predicted, cmap='gray', vmin=0, vmax=1)
            if mask.any():
                axes[n, 2].contour(mask, levels=[.5], colors=['yellow'], linewidths=.6)
            c = row['pixel_counts']
            iou = c['TP'] / max(c['TP'] + c['FP'] + c['FN'], 1)
            axes[n, 2].set_title(f'IoU={iou:.3f}; image pred={row["pred_label"]}')
            errors = np.zeros((*mask.shape, 3), dtype=np.uint8)
            errors[mask & predicted] = (0, 180, 0)
            errors[~mask & predicted] = (255, 0, 0)
            errors[mask & ~predicted] = (0, 100, 255)
            axes[n, 3].imshow(errors)
            axes[n, 3].set_title(f'Green TP / red FP / blue FN\nFP={c["FP"]}, FN={c["FN"]}')
        for axis in axes.flat:
            axis.axis('off')
        fig.suptitle(relative)
        fig.tight_layout()
        dest = output / '可视化' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=100)
        plt.close(fig)
        index.append({'image_path': relative, 'path': dest.relative_to(output).as_posix()})
    write_json(output / 'visualization_index.json', index)
    return index


def run(args, inputs, sources, info, output):
    identities, baseline, manifest, old_cal, old_dev, _ = inputs
    output.mkdir(parents=True, exist_ok=False)
    def stage(name):
        write_json(output / 'status.json', {'status': 'running', 'stage': name, 'at': now()})
    stage('initialize')
    write_json(output / 'preflight.json', info)
    write_json(output / 'split_manifest.json', manifest)
    source_names = ('compare_tiling.py', 'tiling_backend.py', 'compare_resolution.py', 'resolution_metrics.py',
                    'run_trusted_baseline.py', 'baseline_utils.py', 'analyze_thresholds.py')
    config = {'started_at': now(), 'status': 'started_not_completed', 'plan': PLAN,
              'baseline_sources_sha256': identities, 'resolution_sources_sha256': sources,
              'source_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in source_names},
              'manifest_sha256': manifest['manifest_sha256'], 'warmup_each_round': args.warmup, 'repeats': args.repeats,
              'environment': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                              'anomalib': version('anomalib'), 'numpy': version('numpy'),
                              'sklearn': version('scikit-learn'), 'gpu': torch.cuda.get_device_name(0)}}
    write_json(output / 'config.json', config)
    seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model, saved = load_checkpoint(args.baseline_dir / 'model.pt', torch.device('cuda:0'))
    before = state_sha256(model.feature_extractor.state_dict())
    if before != baseline['frozen_backbone_state_sha256'] or saved['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Frozen backbone or checkpoint split mismatch')
    destination = output / 'tiled256'
    destination.mkdir()
    print('[1/4] FIT: native256 overlap tiles, 176 normal images, fixed1802 bank', flush=True)
    stage('build_tiled_bank')
    build = build_bank(model, baseline['config'], manifest, args.data_root, destination)
    if state_sha256(model.feature_extractor.state_dict()) != before:
        raise ValueError('Frozen backbone changed')
    cfg = native_config(baseline['config'])
    cfg['tiling'] = PLAN
    write_json(destination / 'config.json', cfg)
    runner = OverlapTiler(model)
    print('[2/4] CALIBRATE: 44 normal whole images, fused pixel maps and max tile scores', flush=True)
    stage('calibrate')
    threshold, cal_rows = calibrate(runner, cfg, manifest, args.data_root, destination)
    checkpoint = {'format_version': 1, 'model_config': cfg, 'tiling': PLAN,
                  'model_state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                  'manifest_sha256': manifest['manifest_sha256'], 'thresholds': threshold}
    torch.save(checkpoint, destination / 'model.pt')
    del checkpoint
    print('[3/4] EVALUATE: fixed45 development images; held-out excluded', flush=True)
    stage('evaluate')
    tiled_rows, tiled_quality = evaluate(runner, cfg, threshold, manifest, args.data_root, destination)
    # Check actual saved/reloaded model and coordinate fusion, not just serialization.
    sample = manifest['splits']['development'][0]
    image = load_tensor(args.data_root, sample, make_transform(cfg)).unsqueeze(0).to('cuda:0')
    with torch.inference_mode():
        expected = runner(image)
    expected_score = expected.pred_score.cpu()
    expected_map = expected.anomaly_map.cpu()
    del expected, runner, image
    model.to('cpu')
    reloaded, saved_again = load_checkpoint(destination / 'model.pt', torch.device('cuda:0'))
    image = load_tensor(args.data_root, sample, make_transform(cfg)).unsqueeze(0).to('cuda:0')
    reload_runner = OverlapTiler(reloaded)
    with torch.inference_mode():
        observed = reload_runner(image)
    torch.testing.assert_close(observed.pred_score.cpu(), expected_score, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(observed.anomaly_map.cpu(), expected_map, atol=1e-6, rtol=1e-6)
    del observed, image, reload_runner, expected_map, expected_score
    reloaded.to('cpu')
    del reloaded
    roots = {'whole256': args.resolution_dir / '256', 'whole512': args.resolution_dir / '512', 'tiled256': destination}
    rows = {name: load_json(roots[name] / 'development_predictions.json') for name in METHODS[:2]}
    rows['tiled256'] = tiled_rows
    qualities = {name: load_json(roots[name] / 'quality_report.json') for name in METHODS[:2]}
    qualities['tiled256'] = tiled_quality
    thresholds = {name: load_json(roots[name] / 'calibration.json') for name in METHODS[:2]}
    thresholds['tiled256'] = threshold
    pairs = {}
    for name in METHODS[:2]:
        comparison = paired_localization(rows[name], tiled_rows)
        # Historical helper uses resolution suffixes; explicitly relabel every key.
        def relabel(value):
            if isinstance(value, list):
                return [relabel(v) for v in value]
            if isinstance(value, dict):
                return {k.replace('_256', '_reference').replace('_512', '_tiled'): relabel(v) for k, v in value.items()}
            return value
        comparison = relabel(comparison)
        comparison.update({'reference': name, 'candidate': 'tiled256'})
        pairs[name] = comparison
    write_json(output / 'paired_localization.json', pairs)
    whole256, _ = load_checkpoint(args.baseline_dir / 'model.pt', torch.device('cpu'))
    whole512, _ = load_checkpoint(args.resolution_dir / '512/model.pt', torch.device('cpu'))
    models = {'whole256': whole256, 'whole512': whole512, 'tiled256': model}
    state_before = {name: state_sha256(m.state_dict()) for name, m in models.items()}
    configs = {name: copy.deepcopy(baseline['config']) for name in METHODS}
    configs['whole512']['image_size'] = [512, 512]
    configs['tiled256'] = cfg
    print('[4/4] BENCHMARK: fresh three-way cost measurement, cyclic balanced order', flush=True)
    stage('benchmark')
    timings, performance = benchmark(models, configs, manifest, args.data_root, args.warmup, args.repeats)
    write_json(output / 'timings.json', timings)
    if any(state_sha256(m.state_dict()) != state_before[name] for name, m in models.items()):
        raise ValueError('Model state changed during benchmark')
    stage('visualize_and_verify')
    images = visualize(output, roots, rows, thresholds, manifest, args.data_root)
    if any(file_sha256(safe_file(args.baseline_dir, p)) != h for p, h in identities.items()):
        raise ValueError('Frozen baseline source changed')
    if any(file_sha256(safe_file(args.resolution_dir, p)) != h for p, h in sources.items()):
        raise ValueError('Frozen resolution source changed')
    config.update({'status': 'completed', 'completed_at': now()})
    write_json(output / 'config.json', config)
    report = {'status': 'completed_development_overlap_tiling_not_final_test', 'completed_at': now(),
              'config': config, 'build': build, 'qualities': qualities, 'thresholds': thresholds,
              'performance': performance, 'checkpoint': {'sha256': file_sha256(destination / 'model.pt'), 'reload_verified': True},
              'held_out_images_opened': 0, 'source_evidence_unchanged': True, 'new_method_automatically_adopted': False,
              'whole_image_quality_reused_from_verified_saved_resolution_run': True,
              'timing_boundary': 'CUDA-ready normalized input through model and original-coordinate output; tiled includes nine sequential forwards, crop copies and weighted fusion; excludes IO/preprocessing/transfers/metrics',
              'memory_boundary': 'PyTorch peak allocated, one active model; tiled includes original700 input and fusion buffers; not total device memory',
              'limitations': ['One category, one sampling seed, seen development split; not final-test evidence.',
                              'Only3 images below1% GT union area, nonebelow0.1%.',
                              'Tiling changes context, effective receptive fields, original-coordinate smoothing and bank sampling; not an isolated resolution variable.',
                              'Candidate prefilter samples1024/image across9tiles to bound fitting cost; this can discard useful normal features.',
                              'Max-over-tiles image scoring receives its own normal-only threshold.',
                              'Same1802x384 bank storage does not mean equal extraction/inference cost.']}
    write_json(output / 'report.json', report)
    group = '0.1_to_lt_1_percent'
    lines = ['重叠分块对照已完成：开发集45张，70张保留测试未使用。',
             '方案：原图700；256图块，stride224，边缘锚定；9块；重叠加权融合。',
             '176张正常图匹配分块建库；每图均衡抽1024条候选，再用参考coreset保留1802条。',
             '每种方案分别使用44张正常图校准：图像q97.5%，像素q99.5%。',
             '整图质量复用已验证历史结果；三组耗时均在本次重新测量。']
    for name in METHODS:
        q, p = qualities[name], performance[name]
        small = q['area_groups'][group]
        lines.extend([f'{name}: 图像误报={q["image_counts"]["FP"]}/8，漏检={q["image_counts"]["FN"]}/37',
                      f'  异常图平均IoU={q["anomaly_macro"]["pixel_IoU"]:.4f}，AP={q["anomaly_macro"]["pixel_AP"]:.4f}；像素召回={q["pixel_metrics"]["recall"]:.4f}',
                      f'  小面积组n={small["n"]}：平均IoU={small["macro"]["pixel_IoU"]:.4f}，像素召回={small["pixel_metrics"]["recall"]:.4f}',
                      f'  正常像素误报率={q["normal_false_positive_pixels"]["rate"]:.6f}',
                      f'  平均={p["mean_ms"]:.3f}ms，P95={p["p95_ms"]:.3f}ms，分配显存峰值={p["max_peak_allocated_bytes"]/1024**2:.3f}MiB'])
    lines.extend(['没有自动替换当前方案。查看全部45张对照图，绿色=正确缺陷，红色=误圈正常，蓝色=漏掉缺陷。',
                  '同样的正常校准分位数不保证开发集上相同的召回率；不能只根据IoU选择。'])
    (output / '结果摘要.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    import html
    cards = ''.join(f'<details><summary>{html.escape(item["image_path"])}</summary><img loading="lazy" src="{html.escape(item["path"])}"></details>' for item in images)
    (output / '对照报告.html').write_text('<!doctype html><meta charset="utf-8"><title>重叠分块对照</title><style>body{max-width:1300px;margin:30px auto;font:16px sans-serif}pre{white-space:pre-wrap}img{width:100%}summary{padding:12px;cursor:pointer}</style><h1>重叠分块开发集对照</h1><pre>' + html.escape('\n'.join(lines)) + '</pre>' + cards, encoding='utf-8')
    write_json(output / 'status.json', {'status': 'completed', 'completed_at': now()})
    print('\n'.join(lines), flush=True)
    print(f'TILING_COMPARISON_COMPLETED | {output}', flush=True)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=root / '04实验结果/01模型与检索实验/baseline_20261016_204723_088238')
    parser.add_argument('--resolution-dir', type=Path, default=root / '04实验结果/05分辨率对照/resolution_comparison_20261023_161820_426134')
    parser.add_argument('--data-root', type=Path, default=root / '03原始数据/解压数据/MVTecAD')
    parser.add_argument('--manifest', type=Path, default=root / '02代码与配置/configs/splits/metal_nut-v1.json')
    parser.add_argument('--output-root', type=Path, default=root / '04实验结果/08重叠分块对照')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if min(args.warmup, args.repeats) < 1:
        parser.error('Warmup and repeats must be positive')
    for key in ('baseline_dir', 'resolution_dir', 'data_root', 'manifest', 'output_root'):
        setattr(args, key, getattr(args, key).resolve())
    if (not args.output_root.is_relative_to(root / '04实验结果') or
            any(args.output_root == p or args.output_root.is_relative_to(p) for p in (args.baseline_dir, args.resolution_dir, args.data_root))):
        parser.error('Use a new result folder under project04; never frozen sources or raw data')
    inputs, sources, info = preflight(args)
    if args.check_only:
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        print('TILING_PREFLIGHT_PASSED | no fitting or forward', flush=True)
        return
    output = args.output_root / ('tiling_comparison_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f'))
    print(f'OUTPUT: {output}', flush=True)
    try:
        run(args, inputs, sources, info, output)
    except Exception as error:
        if output.exists():
            write_json(output / 'status.json', {'status': 'failed_not_completed', 'at': now(), 'error_type': type(error).__name__, 'error': str(error)})
            (output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
