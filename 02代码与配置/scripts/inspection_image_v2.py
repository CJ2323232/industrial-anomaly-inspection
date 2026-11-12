"""Opt-in v2 image-threshold window; reuse the frozen model and inference code."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
POLICY=ROOT/'02代码与配置/configs/decisions/metal_nut-wrn50-image-v2-candidate.json'

def configure():
    # Reuse the existing inference implementation without editing frozen sources.
    import inspection_runtime as runtime
    import inspection_app as app
    policy=json.loads(POLICY.read_text(encoding='utf-8'))
    if policy['version']!='metal_nut-wrn50-image-v2-candidate':
        raise ValueError('Unexpected candidate version')
    source=runtime.read_json(ROOT/policy['source_policy'])
    if runtime.file_sha256(ROOT/policy['source_policy'])!=policy['source_policy_sha256']:
        raise ValueError('Source policy identity changed')
    for key in ['checkpoint','checkpoint_sha256','backbone','input_size','sigma',
                'feature_layers','memory_bank_shape','num_neighbors','provisional_pixel_threshold']:
        if policy[key]!=source[key]:
            raise ValueError('Image-only candidate changed another model setting: '+key)
    if not policy['previous_image_threshold']==source['image_threshold']:
        raise ValueError('Unexpected source image threshold')
    runtime.DECISION=POLICY
    runtime.VERSION=policy['version']
    app.DECISION=POLICY
    # App and runtime share the same Inspector class and globals.
    inspector=runtime.Inspector()
    if inspector.policy_sha256!=runtime.file_sha256(POLICY):
        raise ValueError('Inspector did not load candidate')
    return runtime,app,inspector

def main():
    runtime,app,inspector=configure()
    if '--check-policy' in sys.argv:
        assert inspector.model is None
        print(json.dumps(dict(version=runtime.VERSION,policy=str(POLICY),
            image_threshold=inspector.policy['image_threshold'],
            pixel_threshold=inspector.policy['provisional_pixel_threshold'],
            model_loaded=False,model_forwards=0),ensure_ascii=False))
        return
    if '--image' in sys.argv or '--preflight' in sys.argv:
        app.main()
        return
    root=app.tk.Tk()
    app.InspectionApp(root)
    root.title('工业视觉异常检测 · WRN50 图像阈值 v2 候选')
    app.ttk.Label(root,text='v2：图像阈值由开发集标签选择，独立效果待验证。',
        font=('Microsoft YaHei UI',10)).pack(side='bottom',pady=4)
    root.mainloop()

if __name__=='__main__': main()
