"""Read-only existing-development-score scan, postprocessing audit and selection.

No fitting, forward pass, architecture edits or held-out data access. Writes a
new timestamped directory. The prespecified selection criterion is maximum
pooled Pixel IoU at the frozen ResNet18 pooled recall, among saved configurations.
Costs and per-image regressions are reported separately, without a hidden score.
"""
import argparse
from decimal import Decimal, ROUND_CEILING
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .analyze_matched_recall import ROOT, SOURCE, DATA, NAMES, now, read, write, load_inputs
    from .baseline_utils import file_sha256, load_manifest, safe_file
    from .matched_recall_metrics import counts_at, sorted_scores
    from .resolution_metrics import evaluate_records
    from .threshold_scan_metrics import pixel_metrics, rank_metrics, cutoff_at_count, oracle_image_iou
else:
    from analyze_matched_recall import ROOT, SOURCE, DATA, NAMES, now, read, write, load_inputs
    from baseline_utils import file_sha256, load_manifest, safe_file
    from matched_recall_metrics import counts_at, sorted_scores
    from resolution_metrics import evaluate_records
    from threshold_scan_metrics import pixel_metrics, rank_metrics, cutoff_at_count, oracle_image_iou

VARIANTS = {
    'resnet18_512': ('05分辨率对照/resolution_comparison_20261023_161820_426134', '512'),
    'resnet18_sigma2': ('06平滑强度对照/smoothing_comparison_20261023_202433_801686', 'sigma2'),
    'resnet18_tiled': ('08重叠分块对照/tiling_comparison_20261024_235337_638241', 'tiled256'),
}
LABELS = {'resnet18': 'ResNet18 / 256 / sigma4（基线）',
          'wide_resnet50': 'WideResNet50 / 256 / sigma4',
          'resnet18_512': 'ResNet18 / 512 / sigma4',
          'resnet18_sigma2': 'ResNet18 / 256 / sigma2',
          'resnet18_tiled': 'ResNet18 / 256重叠分块 / sigma4'}


def load_variant(name, relative, key, manifest, masks, data):
    root = ROOT / '04实验结果' / relative
    report = read(root / 'report.json')
    if report['held_out_images_opened'] != 0 or not report['status'].startswith('completed'):
        raise ValueError('Require completed development-only source: ' + name)
    if load_manifest(root / 'split_manifest.json')['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Different split: ' + name)
    directory = root / key
    rows = read(directory / 'development_predictions.json')
    paths = manifest['splits']['development']
    if len(rows) != len(paths) or {r['image_path'] for r in rows} != set(paths):
        raise ValueError('Different development image list: ' + name)
    indexed = {r['image_path']: r for r in rows}
    scores, hashes = {}, {}
    for path in paths:
        row, rec = indexed[path], manifest['files'][path]
        for field in ('image_sha256', 'mask_sha256', 'label', 'defect_type'):
            if row[field] != rec[field]:
                raise ValueError('Metadata mismatch: ' + name + ':' + path)
        expected = 'maps/development/' + Path(path).with_suffix('.npz').as_posix()
        if row['map_path'] != expected:
            raise ValueError('Only development maps allowed')
        file = safe_file(directory, expected)
        hashes[file.relative_to(ROOT).as_posix()] = file_sha256(file)
        with np.load(file, allow_pickle=False) as z:
            a = z['anomaly_map']
        if a.dtype != np.float32 or a.shape != masks[path].shape or not np.isfinite(a).all():
            raise ValueError('Invalid continuous map: ' + name + ':' + path)
        if len(np.unique(a.reshape(-1)[:4096])) <= 2 and len(np.unique(a)) <= 2:
            raise ValueError('Binary or degenerate map cannot establish continuous scores')
        scores[path] = (sorted_scores(a[masks[path]]), sorted_scores(a[~masks[path]]))
        if counts_at(*scores[path], report['thresholds'][key]['pixel_threshold']) != row['pixel_counts']:
            raise ValueError('Original threshold counts do not reproduce')
    if evaluate_records(rows) != {k: report['qualities'][key][k] for k in evaluate_records(rows)}:
        raise ValueError('Original quality does not reproduce')
    for path in ('report.json', 'config.json', 'split_manifest.json', key + '/development_predictions.json'):
        f = root / path
        hashes[f.relative_to(ROOT).as_posix()] = file_sha256(f)
    model = {'rows': indexed, 'scores': scores,
             'positive_sorted': np.sort(np.concatenate([v[0] for v in scores.values()]))}
    return model, directory, report, key, hashes


def evaluate(model, paths, threshold):
    rows = []
    for path in paths:
        base = model['rows'][path]
        counts = counts_at(*model['scores'][path], threshold)
        rows.append({'image_path': path, 'label': base['label'], 'defect_type': base['defect_type'],
                     'area_ratio': base['area_ratio'], **pixel_metrics(counts)})
    total = {k: sum(r[k] for r in rows) for k in ('TP', 'FP', 'FN', 'TN')}
    anomalies = [r for r in rows if r['label']]
    small = [r for r in anomalies if r['area_ratio'] < .01]
    normal = [r for r in rows if not r['label']]
    small_counts = {k: sum(r[k] for r in small) for k in total}
    summary = {'threshold': float(np.float32(threshold)), **pixel_metrics(total),
               'macro_IoU_37_anomalies': float(np.mean([r['Pixel_IoU'] for r in anomalies])),
               'macro_Recall_37_anomalies': float(np.mean([r['Pixel_Recall'] for r in anomalies])),
               'small_n': len(small), 'small_macro_IoU': float(np.mean([r['Pixel_IoU'] for r in small])) if small else None,
               'small_pixel_recall': pixel_metrics(small_counts)['Pixel_Recall'],
               'normal_images_FP': sum(r['FP'] for r in normal),
               'normal_images_FP_fraction': sum(r['FP'] for r in normal) / sum(sum(r[k] for k in total) for r in normal)}
    return summary, rows


def audit_pipeline(source_report, directories):
    """Inspect saved configuration, recorded source identities and CPU checkpoint buffers."""
    import torch
    source_hashes = source_report['config']['source_sha256']
    source_matches = {f: file_sha256(ROOT / '02代码与配置/scripts' / f) == digest for f, digest in source_hashes.items()}
    configs = {name: read(directories[name] / 'config.json') for name in NAMES}
    equal_fields = ('implementation', 'layers', 'num_neighbors', 'image_size', 'mean', 'std',
                    'resize_antialias', 'center_crop', 'reference_map_upsampling',
                    'original_coordinate_mapping', 'eval_batch_size', 'score_normalization', 'decision_rule')
    comparisons = {f: {'resnet18': configs['resnet18'][f], 'wide_resnet50': configs['wide_resnet50'][f],
                        'same': configs['resnet18'][f] == configs['wide_resnet50'][f]} for f in equal_fields}
    kernels, checkpoints = {}, {}
    for name in NAMES:
        path = directories[name] / 'model.pt'
        digest = file_sha256(path)
        if digest != source_report['checkpoints'][name]['sha256']:
            raise ValueError('Checkpoint identity mismatch')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['model_config'] != configs[name]:
            raise ValueError('Checkpoint and saved config differ')
        kernel = saved['model_state_dict']['anomaly_map_generator.blur.kernel'].numpy().copy()
        kernels[name] = kernel
        checkpoints[name] = {'sha256': digest, 'kernel_shape': list(kernel.shape),
                             'bank_shape': list(saved['model_state_dict']['memory_bank'].shape)}
        del saved
    kernel_equal = np.array_equal(kernels[NAMES[0]], kernels[NAMES[1]])
    x = np.arange(-16, 17, dtype=np.float64)
    g = np.exp(-x*x / (2*4**2)); g /= g.sum()
    expected = np.outer(g, g)[None, None]
    errors = {n: float(np.max(np.abs(k - expected))) for n, k in kernels.items()}
    site = ROOT / '.venv/Lib/site-packages'
    library_paths = ['anomalib/models/image/patchcore/torch_model.py', 'anomalib/models/image/patchcore/anomaly_map.py',
                     'anomalib/models/components/filters/blur.py']
    library = {p: file_sha256(site / p) for p in library_paths}
    same = all(v['same'] for v in comparisons.values()) and all(source_matches.values()) and kernel_equal and max(errors.values()) < 1e-7
    return {'backbone_pair_same_configured_preprocessing_and_postprocessing': same,
            'config_comparisons': comparisons, 'recorded_project_source_hashes_match': source_matches,
            'checkpoint_metadata_and_hashes_verified': checkpoints, 'stored_gaussian_kernels_identical': kernel_equal,
            'sigma4_33x33_analytic_kernel_max_errors': errors, 'current_library_source_sha256': library,
            'current_anomalib': version('anomalib'), 'recorded_anomalib': source_report['config']['environment']['anomalib'],
            'steps': ['RGB -> float32 /255 -> antialiased resize256 -> identical ImageNet mean/std',
                      'layer2/layer3 -> AvgPool3,stride1,pad1 -> bilinear feature alignment -> channel concatenation',
                      'reference nearest-neighbor raw Euclidean pixel distances; no score normalization',
                      '32x32 patch map -> nearest upsample256 -> Gaussian sigma4,33x33,reflect,same',
                      'bilinear upsample700x700,align_corners=False -> saved float32 raw map',
                      'GT original700x700 binary0/255; no GT resampling, crop or rotation',
                      'scan applies strict > only; no morphology, components, adaptive thresholds or filtering'],
            'intentional_variant_differences': {'resnet18_512': '512 input,64x64 map; other settings retained',
                'resnet18_sigma2': 'same256/reference bank; Gaussian sigma2 instead of4',
                'resnet18_tiled': '9 overlapping native256 crops; triangular weighted merge; normal tiled bank'},
            'limitations': ['Original pre-smoothing patch distances were not retained; cannot replay historical smoothing from upstream maps.',
                            'Library files were not individually hashed in the original run; current2.2.0 implementation, recorded project code and saved kernels support this audit.',
                            'Backbone comparison also changes ImageNet checkpoint and feature dimension384->1536; improvements cannot be attributed to architecture alone.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quantile-points', type=int, default=501)
    args = parser.parse_args()
    if args.quantile_points < 3:
        parser.error('Need at least3 score quantiles')
    output = ROOT / '04实验结果/11阈值扫描与方案选择' / ('threshold_scan_' + now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'status.json', {'status': 'running', 'started_at': now().isoformat()})
    protocol = {'only_existing_development_data': True, 'new_forward_passes': 0, 'new_training': False,
        'raw_threshold_scan': f'{args.quantile_points} empirical score quantiles across all pixels plus extrema and original cutoff; equal quantile spacing, not equal raw-score spacing',
        'common_recall_grid': '0..100% with0.1 percentage-point target spacing; empirical tied values included whole',
        'primary_selection': 'largest pooled Pixel IoU at frozen ResNet18 pooled Pixel Recall among the5 stored configurations',
        'secondary_sensitivity': 'same minimum pooled recall AND100% recall on the3 known small-area images',
        'tie_rule': 'one global float32 cutoff per method; strict score > cutoff; no pixel randomization or interpolation of metrics',
        'GT_scope': '45 development images:37 anomalous,8 normal;70 held-out images never opened',
        'deployment_calibration': False, 'automatic_adoption': False}
    write(output / 'protocol.json', protocol)
    manifest, source_report, models, masks, identities, map_hashes, original_quality = load_inputs(SOURCE, DATA)
    paths = manifest['splits']['development']
    directories = {n: SOURCE / n for n in NAMES}
    sources = {n: (source_report, n) for n in NAMES}
    frozen_hashes = {(SOURCE / p).relative_to(ROOT).as_posix(): h for p, h in (identities | map_hashes).items()}
    for name, (relative, key) in VARIANTS.items():
        model, directory, report, key, hashes = load_variant(name, relative, key, manifest, masks, DATA)
        models[name], directories[name], sources[name] = model, directory, (report, key)
        frozen_hashes.update(hashes)
        print('loaded saved continuous maps: ' + name, flush=True)
    pipeline = audit_pipeline(source_report, directories)
    write(output / 'pipeline_audit.json', pipeline)
    if not pipeline['backbone_pair_same_configured_preprocessing_and_postprocessing']:
        raise ValueError('Backbone pipeline audit differs; inspect before interpreting scores')
    inventory, ranking, raw_scans, recall_scans, matched, frozen, small_guard = {}, {}, {}, {}, {}, {}, {}
    baseline_counts = original_quality['resnet18']['pixel_counts']
    pcount = baseline_counts['TP'] + baseline_counts['FN']
    required_anchor = baseline_counts['TP']
    targets = [('recall90', int(np.ceil(.9*pcount))), ('recall95', int(np.ceil(.95*pcount))),
               ('baseline_recall', required_anchor), ('recall97', int(np.ceil(.97*pcount))),
               ('recall99', int(np.ceil(.99*pcount)))]
    for name, model in models.items():
        positive = model['positive_sorted']
        negative = np.sort(np.concatenate([v[1] for v in model['scores'].values()]))
        if len(positive) != pcount:
            raise ValueError('Methods have different GT pixel populations')
        all_values = np.concatenate((positive, negative))
        empirical_quantiles = np.quantile(all_values, np.linspace(0, 1, args.quantile_points), method='linear').astype(np.float32)
        source, key = sources[name]
        original_threshold = source['thresholds'][key]['pixel_threshold']
        thresholds = np.unique(np.r_[empirical_quantiles,
                              np.nextafter(all_values.min(), np.float32(-np.inf)), np.float32(original_threshold)])
        inventory[name] = {'label': LABELS[name], 'source': directories[name].relative_to(ROOT).as_posix(),
            'development_images': len(paths), 'GT_positive_pixels': len(positive), 'GT_negative_pixels': len(negative),
            'dtype': 'float32', 'original_shape': [700, 700], 'score_min': float(all_values.min()),
            'score_max': float(all_values.max()), 'score_unique_sample_count': int(len(np.unique(all_values[:8192]))),
            'continuous_not_binary': len(np.unique(all_values[:8192])) > 2, 'raw_threshold_count': len(thresholds)}
        del all_values
        ranking[name] = {'pooled': rank_metrics(positive, negative),
                         'per_image': {path: rank_metrics(*model['scores'][path]) | {'oracle': oracle_image_iou(*model['scores'][path])} for path in paths}}
        anomalies = [r for path, r in ranking[name]['per_image'].items() if manifest['files'][path]['label']]
        ranking[name]['macro_AP_37'] = float(np.mean([r['AP'] for r in anomalies]))
        ranking[name]['macro_AUROC_37'] = float(np.mean([r['AUROC'] for r in anomalies]))
        ranking[name]['oracle_macro_IoU_37'] = float(np.mean([r['oracle']['max_IoU'] for r in anomalies]))
        ranking[name]['oracle_IoU_below_05_images'] = sum(r['oracle']['max_IoU'] < .5 for r in anomalies)
        if not np.isclose(ranking[name]['macro_AP_37'], source['qualities'][key]['anomaly_macro']['pixel_AP'], atol=1e-12):
            raise ValueError('Recomputed AP differs from original source')
        raw_scans[name] = []
        with (output / (name + '_逐阈值逐图.jsonl')).open('w', encoding='utf-8') as stream:
            for i, tau in enumerate(thresholds):
                summary, rows = evaluate(model, paths, float(tau))
                raw_scans[name].append(summary)
                for row in rows:
                    stream.write(json.dumps({'threshold_index': i, 'threshold': float(tau), **row}, ensure_ascii=False, allow_nan=False) + '\n')
        recall_scans[name] = []
        for i in range(1001):
            required_tp = int((Decimal(i) * pcount / 1000).to_integral_value(rounding=ROUND_CEILING))
            tau = cutoff_at_count(positive, negative, required_tp)
            summary, _ = evaluate(model, paths, tau)
            summary.update({'target_recall': i / 1000, 'required_TP': required_tp,
                            'extra_TP_due_to_ties': summary['TP'] - required_tp})
            recall_scans[name].append(summary)
        matched[name] = {}
        for label, required_tp in targets:
            tau = cutoff_at_count(positive, negative, required_tp)
            summary, rows = evaluate(model, paths, tau)
            matched[name][label] = {'summary': summary, 'per_image': rows, 'required_TP': required_tp,
                                   'target_recall': required_tp / pcount, 'extra_TP_due_to_ties': summary['TP'] - required_tp}
        frozen[name] = dict(zip(('summary', 'per_image'), evaluate(model, paths, original_threshold)))
        tau_anchor = matched[name]['baseline_recall']['summary']['threshold']
        small_min = min(float(model['scores'][path][0][0]) for path in paths
                        if model['rows'][path]['label'] and model['rows'][path]['area_ratio'] < .01)
        guard_tau = min(tau_anchor, float(np.nextafter(np.float32(small_min), np.float32(-np.inf))))
        summary, rows = evaluate(model, paths, guard_tau)
        small_guard[name] = {'summary': summary, 'per_image': rows}
        assert summary['TP'] >= required_anchor and summary['small_pixel_recall'] == 1.
        print(name + ' anchor: ' + json.dumps(matched[name]['baseline_recall']['summary'], ensure_ascii=False), flush=True)
        del negative
    write(output / 'data_inventory.json', inventory)
    write(output / 'ranking_metrics.json', ranking)
    write(output / 'raw_threshold_scan.json', raw_scans)
    write(output / 'common_recall_scan.json', recall_scans)
    write(output / 'matched_working_points.json', matched)
    write(output / 'original_working_points.json', frozen)
    write(output / 'small_defect_guard_points.json', small_guard)
    best = max(models, key=lambda n: matched[n]['baseline_recall']['summary']['Pixel_IoU'])
    guard_best = max(models, key=lambda n: small_guard[n]['summary']['Pixel_IoU'])
    selection = {'primary_rule': protocol['primary_selection'], 'primary_winner': best,
                 'small_defect_guard_rule': protocol['secondary_sensitivity'], 'small_guard_winner': guard_best,
                 'scope': 'empirical saved-development candidates only; no deployment guarantee or automatic switch'}
    write(output / 'selection.json', selection)
    # Independent direct-count checks from score maps, rather than sorted ranks.
    direct_checks = 0
    for name, directory in directories.items():
        for path in paths:
            with np.load(safe_file(directory, models[name]['rows'][path]['map_path']), allow_pickle=False) as z:
                array = z['anomaly_map']
            mask = masks[path]
            for point in matched[name].values():
                tau = point['summary']['threshold']
                pred = array > tau
                actual = {'TP': int((pred & mask).sum()), 'FP': int((pred & ~mask).sum()),
                          'FN': int((~pred & mask).sum()), 'TN': int((~pred & ~mask).sum())}
                row = next(r for r in point['per_image'] if r['image_path'] == path)
                if actual != {k: row[k] for k in actual}:
                    raise ValueError('Direct score-mask statistics differ')
                direct_checks += 1
    for relative, digest in frozen_hashes.items():
        if file_sha256(ROOT / relative) != digest:
            raise ValueError('Original experimental evidence changed')
    if __package__:
        from .threshold_scan_reporting import generate_report
    else:
        from threshold_scan_reporting import generate_report
    generate_report(output, models, directories, masks, DATA, LABELS, inventory, pipeline,
                    ranking, raw_scans, recall_scans, matched, frozen, small_guard, selection, sources)
    verification = {'frozen_source_files_sha256': frozen_hashes, 'source_evidence_unchanged': True,
                    'source_code_sha256': {f: file_sha256(Path(__file__).with_name(f)) for f in
                                         ('run_threshold_scan.py', 'threshold_scan_metrics.py', 'threshold_scan_reporting.py')},
                    'direct_map_counts_checked': direct_checks, 'held_out_images_opened': 0,
                    'continuous_maps_loaded': len(models)*len(paths), 'GT_masks_loaded': sum(manifest['files'][p]['label'] for p in paths),
                    'numpy_version': np.__version__}
    write(output / 'verification.json', verification)
    write(output / 'status.json', {'status': 'completed', 'completed_at': now().isoformat(), 'new_training': False, 'new_inference': False})
    print('THRESHOLD_SCAN_COMPLETED | ' + str(output), flush=True)


if __name__ == '__main__':
    main()
