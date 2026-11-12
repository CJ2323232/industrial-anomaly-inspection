"""Offline single-image inference using the selected existing PatchCore checkpoint."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for key, rel in {'TORCH_HOME': '.cache/models/torch', 'HF_HOME': '.cache/models/huggingface',
                 'XDG_CACHE_HOME': '.cache/models', 'MPLCONFIGDIR': '.cache/matplotlib',
                 'TEMP': '.cache/tmp', 'TMP': '.cache/tmp'}.items():
    os.environ[key] = str(ROOT / rel)
    (ROOT / rel).mkdir(parents=True, exist_ok=True)
os.environ['HF_HUB_OFFLINE'] = '1'
import hashlib
import html
import io
import json
import platform
import time
from datetime import datetime, timezone, timedelta
import numpy as np
from PIL import Image
if __package__:
    from .baseline_utils import load_manifest, file_sha256, safe_file
else:
    from baseline_utils import load_manifest, file_sha256, safe_file
DECISION = ROOT / '02代码与配置/configs/decisions/metal_nut-wrn50-selected-v1.json'
MANIFEST = ROOT / '02代码与配置/configs/splits/metal_nut-v1.json'
DATA = ROOT / '03原始数据/解压数据/MVTecAD'
OUTPUT = ROOT / '04实验结果/14单图检测'
VERSION = 'wrn50-inspector-v1'
MAX_BYTES = 40 * 1024 * 1024
MAX_PIXELS = 16_000_000

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

class InputGuard:
    """Reject known held-out paths BEFORE reading, and renamed copies by manifest hash."""
    def __init__(self, manifest=None, data_root=DATA):
        self.manifest = manifest or load_manifest(MANIFEST)
        self.data_root = Path(data_root).resolve()
        self.sealed_paths = {(self.data_root / p).resolve() for p in self.manifest['splits']['held_out']}
        self.sealed_hashes = {self.manifest['files'][p]['image_sha256'] for p in self.manifest['splits']['held_out']}
        self.by_hash = {r['image_sha256']: p for p,r in self.manifest['files'].items()}
        self.split_by_path = {p: s for s,paths in self.manifest['splits'].items() for p in paths}

    def load(self, path):
        path = Path(path).expanduser().resolve()
        if path in self.sealed_paths:
            raise ValueError('这是保留测试图，当前入口不开放。请选择开发集样例或你自己的新图片。')
        if path.is_relative_to(self.data_root):
            relative = path.relative_to(self.data_root).as_posix()
            if relative not in self.manifest['files']:
                raise ValueError('原始数据目录只接受已登记的非保留图片，不能选择GT掩码。')
        if not path.is_file() or path.suffix.lower() not in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}:
            raise ValueError('请选择PNG、JPG、BMP或TIFF图片。')
        if path.stat().st_size > MAX_BYTES:
            raise ValueError('图片文件超过40MB，请先缩小文件。')
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest in self.sealed_hashes:
            raise ValueError('这张图片与保留测试图内容一致，当前入口不开放。')
        known = self.by_hash.get(digest)
        with Image.open(io.BytesIO(content)) as im:
            if im.width * im.height > MAX_PIXELS:
                raise ValueError('图片超过1600万像素，请先缩小尺寸。')
            if getattr(im, 'n_frames', 1) != 1:
                raise ValueError('请使用单帧图片。')
            mode = im.mode
            image = im.convert('RGB')
            image.load()
        return image, {'selected_path': str(path), 'sha256': digest, 'source_mode': mode,
            'manifest_image': known, 'split': self.split_by_path.get(known, 'user_image'),
            'width': image.width, 'height': image.height, 'exif_rotation_applied': False}

def threshold_result(score, score_map, image_threshold, pixel_threshold):
    if not np.isfinite(score) or not np.isfinite(score_map).all() or score_map.ndim != 2:
        raise ValueError('模型返回了无效分数。')
    return bool(score > image_threshold), score_map > pixel_threshold

def visualizations(image, score_map, pixel_threshold):
    """Display normalization never changes the prediction or stored raw map."""
    from matplotlib import colormaps
    normalized = np.clip(score_map / (2 * pixel_threshold), 0, 1)
    heat = np.round(colormaps['turbo'](normalized)[..., :3] * 255).astype(np.uint8)
    mask = score_map > pixel_threshold
    original = np.asarray(image, dtype=np.uint8)
    overlay = original.copy()
    overlay[mask] = np.round(.55 * original[mask] + .45 * np.array([255, 50, 45])).astype(np.uint8)
    return Image.fromarray(heat), Image.fromarray(overlay), Image.fromarray(mask.astype(np.uint8)*255)

class Inspector:
    def __init__(self, device='auto'):
        self.guard = InputGuard()
        self.policy = read_json(DECISION)
        self.policy_sha256 = file_sha256(DECISION)
        self.checkpoint = safe_file(ROOT, self.policy['checkpoint'])
        if file_sha256(self.checkpoint) != self.policy['checkpoint_sha256']:
            raise ValueError('模型文件校验不一致，停止加载。')
        self.device_request = device
        self.model = None
        self.model_load_ms = None
        self.config = read_json(self.checkpoint.parent/'config.json')
        if self.config['backbone'] != self.policy['backbone'] or self.config['image_size'] != self.policy['input_size']:
            raise ValueError('选型记录与模型配置不一致。')

    def load_model(self):
        if self.model is not None: return
        start = time.perf_counter()
        import torch
        if __package__:
            from .run_trusted_baseline import load_checkpoint, make_transform
        else:
            from run_trusted_baseline import load_checkpoint, make_transform
        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if self.device_request == 'auto' else self.device_request)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA不可用，请使用自动或CPU模式。')
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        model, saved = load_checkpoint(self.checkpoint, self.device)
        model.requires_grad_(False)
        if saved['manifest_sha256'] != self.guard.manifest['manifest_sha256']:
            raise ValueError('模型数据划分与当前记录不一致。')
        if saved['model_config'] != self.config or list(model.memory_bank.shape) != self.policy['memory_bank_shape']:
            raise ValueError('模型配置或正常特征库形状不一致。')
        self.transform = make_transform(self.config)
        self.model = model
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        self.model_load_ms = (time.perf_counter()-start)*1000

    def infer(self, path):
        total_start = time.perf_counter()
        # Validate before loading the expensive model; use decoded bytes throughout.
        input_start = time.perf_counter()
        image, identity = self.guard.load(path)
        decode_ms = (time.perf_counter()-input_start)*1000
        self.load_model()
        start = time.perf_counter()
        import torch
        if __package__:
            from .run_trusted_baseline import predict
        else:
            from run_trusted_baseline import predict
        tensor = self.transform(image).to(self.device)
        score, score_map = predict(self.model, tensor, (image.height, image.width))
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        inference_ms = decode_ms + (time.perf_counter()-start)*1000
        it = self.policy['image_threshold']
        pt = self.policy['provisional_pixel_threshold']
        anomalous, mask = threshold_result(score, score_map, it, pt)
        timestamp = datetime.now(timezone(timedelta(hours=8)))
        output = OUTPUT / timestamp.strftime('inspection_%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True, exist_ok=False)
        save_json(output/'status.json', {'status':'exporting'})
        heat, overlay, mask_image = visualizations(image, score_map, pt)
        image.save(output/'原图.png'); heat.save(output/'热力图.png')
        overlay.save(output/'缺陷叠加.png'); mask_image.save(output/'预测掩码.png')
        np.savez_compressed(output/'连续异常分数.npz', anomaly_map=score_map)
        result = {'version': VERSION, 'created_at': timestamp.isoformat(), 'input': identity,
            'model': {'checkpoint': self.policy['checkpoint'], 'checkpoint_sha256': self.policy['checkpoint_sha256'],
                'decision_sha256': self.policy_sha256, 'backbone': self.policy['backbone'],
                'input_size': self.policy['input_size'], 'sigma': self.policy['sigma'],
                'memory_bank_shape': self.policy['memory_bank_shape'], 'device': str(self.device)},
            'raw_image_score': score, 'image_threshold': it, 'image_is_anomalous': anomalous,
            'pixel_threshold': pt, 'pixel_threshold_status': self.policy['pixel_threshold_status'],
            'decision_rule': 'raw score > threshold', 'predicted_pixels': int(mask.sum()),
            'predicted_area_fraction': float(mask.mean()), 'score_map_min': float(score_map.min()),
            'score_map_max': float(score_map.max()), 'heatmap_display_range': [0, 2*pt],
            'heatmap_changes_prediction': False,
            'timing': {'decode_preprocess_transfer_infer_restore_ms': inference_ms,
                'model_load_ms_this_session': self.model_load_ms,
                'request_through_image_export_ms': (time.perf_counter()-total_start)*1000,
                'notes': '单次交互耗时，不是稳态基准；检测计时含校验/解码/预处理/传输/前向/原尺寸还原，排除模型加载和文件导出。'},
            'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda},
            'training_performed': False, 'held_out_evaluation': False,
            'source_sha256': {Path(__file__).name: file_sha256(__file__), 'run_trusted_baseline.py': file_sha256(Path(__file__).with_name('run_trusted_baseline.py'))}}
        save_json(output/'结果.json', result)
        write_report(output, result)
        save_json(output/'status.json', {'status':'completed'})
        return output, result

def write_report(output, r):
    decision='检测为异常' if r['image_is_anomalous'] else '整图未判异常'
    figures=''.join(f'<figure><img src="{name}.png"><figcaption>{label}</figcaption></figure>' for name,label in [('原图','输入原图'),('缺陷叠加','红色：超过像素阈值的候选区域'),('热力图','固定色标热力图：蓝低、红高；不是概率'),('预测掩码','白色：预测候选区域')])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>单图检测结果</title>
<style>body{{max-width:1150px;margin:35px auto;padding:0 22px;background:#f4f7fb;color:#193348;font:16px/1.7 system-ui}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure{{background:white;padding:15px;margin:0;border-radius:12px}}img{{width:100%;max-height:530px;object-fit:contain}}.stats{{padding:20px;background:white;border-radius:12px;margin:20px 0}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>{decision}</h1><p>PatchCore · WideResNet50 · 256 × 256 · sigma 4 · 开发候选版本</p>
<div class="stats">图片：{html.escape(r['input']['manifest_image'] or Path(r['input']['selected_path']).name)}<br>
整图原始分数 <b>{r['raw_image_score']:.6f}</b> / 阈值 {r['image_threshold']:.6f}<br>
像素阈值 {r['pixel_threshold']:.6f} / 候选区域 {r['predicted_pixels']:,} 像素（{r['predicted_area_fraction']:.2%}）<br>
本次检测 {r['timing']['decode_preprocess_transfer_infer_restore_ms']:.1f} ms（含解码、预处理、传输和模型处理，不含模型加载及导出；单次耗时）</div>
<p>整图判定与像素定位使用不同分数和阈值，因此可能出现整图未判异常但仍有少量候选区域。单张新图片没有人工GT，不显示IoU等准确率。</p>
<div class="grid">{figures}</div><p>像素阈值用开发集标注选择，尚未完成独立最终验证。适用对象为本项目金属螺母；其它类别输出不能当作可靠判断。热力图固定显示范围0—{r['heatmap_display_range'][1]:.4f}，超过上限的颜色饱和；原始连续分数另行保存。</p>
<p><a href="结果.json">配置与原始指标</a> · <a href="连续异常分数.npz">连续分数文件</a></p></html>'''
    (output/'检测报告.html').write_text(page,encoding='utf-8')
