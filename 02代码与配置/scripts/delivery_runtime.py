"""Versioned delivery wrapper. Original frozen algorithms and runtime are unchanged."""
from pathlib import Path
import argparse,html,json,time,logging
from datetime import datetime
import numpy as np
from inspection_runtime import ROOT,InputGuard,Inspector,file_sha256,read_json,save_json,visualizations,threshold_result
PROFILES=ROOT/'02代码与配置/configs/delivery/final_demo_profiles.json'
class DeliveryInspector(Inspector):
    def __init__(self,profile='r18_reference',device='auto',output_dir=None):
        table=read_json(PROFILES)
        if profile not in table['profiles']:raise ValueError('未知版本：'+profile)
        self.profile=profile;self.policy=table['profiles'][profile]
        self.policy_sha256=file_sha256(PROFILES);self.guard=InputGuard()
        self.checkpoint=(ROOT/self.policy['checkpoint']).resolve()
        if not self.checkpoint.is_relative_to(ROOT):raise ValueError('权重路径必须位于项目内。')
        if file_sha256(self.checkpoint)!=self.policy['checkpoint_sha256']:raise ValueError('权重SHA256不一致，拒绝运行。')
        self.config=read_json(self.checkpoint.parent/'config.json')
        if self.config['backbone']!=self.policy['backbone'] or self.config['image_size']!=self.policy['input_size']:raise ValueError('版本与权重配置不一致。')
        self.device_request=device;self.model=None;self.model_load_ms=None
        self.output_base=Path(output_dir or ROOT/'04实验结果/26最终演示').resolve()
        if not self.output_base.is_relative_to((ROOT/'04实验结果').resolve()):raise ValueError('输出目录必须在项目04实验结果内。')
    def infer(self,path):
        start=time.perf_counter();image,identity=self.guard.load(path);self.load_model()
        from run_trusted_baseline import predict
        import torch
        captured=[];handle=None
        if self.policy.get('scoring_method')=='topk_mean':
            # Capture the unchanged input to the existing map generator, before any smoothing.
            def capture_patch_distances(module,args):
                captured.append(args[0].detach().cpu().numpy().copy())
            handle=self.model.anomaly_map_generator.register_forward_pre_hook(capture_patch_distances)
        try:
            score,amap=predict(self.model,self.transform(image).to(self.device),(image.height,image.width))
        finally:
            if handle is not None:handle.remove()
        scoring_details=None
        if self.policy.get('scoring_method')=='topk_mean':
            from delivery_topk import topk_image_score
            if len(captured)!=1:raise ValueError('C评分没有取得唯一的原始Patch距离图。')
            original_A_score=score
            score,scoring_details=topk_image_score(captured[0],self.policy['K'],self.policy['expected_patch_count'])
            scoring_details['reference_A_score']=original_A_score
        it=self.policy['image_threshold'];pt=self.policy['provisional_pixel_threshold']
        abnormal,mask=threshold_result(score,amap,it,pt)
        out=self.output_base/datetime.now().strftime('demo_%Y%m%d_%H%M%S_%f')
        out.mkdir(parents=True,exist_ok=False)
        try:
            save_json(out/'status.json',{'status':'exporting'})
            heat,overlay,mask_image=visualizations(image,amap,pt)
            for name,img in [('原图',image),('热力图',heat),('缺陷叠加',overlay),('预测掩码',mask_image)]:img.save(out/(name+'.png'))
            np.savez_compressed(out/'连续异常分数.npz',anomaly_map=amap)
            if captured:np.savez_compressed(out/'Patch异常距离.npz',patch_distances=captured[0].reshape(32,32))
            result=dict(profile=self.profile,profile_name=self.policy['name'],profile_sha256=self.policy_sha256,
                checkpoint=self.policy['checkpoint'],checkpoint_sha256=self.policy['checkpoint_sha256'],
                model_config=self.config,memory_bank_shape=self.policy['memory_bank_shape'],
                input=identity,raw_image_score=score,image_threshold=it,pixel_threshold=pt,
                image_is_anomalous=abnormal,predicted_pixels=int(mask.sum()),predicted_area_fraction=float(mask.mean()),
                image_rule='raw_score > image_threshold',pixel_rule='raw_original_resolution_map > pixel_threshold',
                scoring=self.policy['scoring'],scoring_details=scoring_details,experimental_candidate=self.policy.get('experimental_candidate',False),evidence_limit=self.policy.get('evidence_limit'),heatmap_display_range=[0,2*pt],pixel_mask_gated_by_image=False,
                note='演示输出不是准确率评估；整图判断与像素掩码独立。跨产品数据不能当作可靠检测。历史70图默认入口拒绝访问。',
                training_performed=False,elapsed_including_load_export_ms=(time.perf_counter()-start)*1000,
                source_sha256={f.name:file_sha256(f) for f in [Path(__file__),Path(__file__).with_name('run_trusted_baseline.py'),Path(__file__).with_name('inspection_runtime.py'),Path(__file__).with_name('delivery_topk.py')]})
            save_json(out/'结果.json',result)
            figures=''.join('<figure><img src="'+n+'.png"><figcaption>'+n+'</figcaption></figure>' for n in ['原图','热力图','缺陷叠加','预测掩码'])
            title='整图检测为异常' if abnormal else '整图未判异常'
            page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>最终演示</title><style>body{max-width:1100px;margin:30px auto;font:16px/1.8 system-ui;background:#f5f7fb;color:#173345;padding:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{background:white;margin:0;padding:10px}img{width:100%;max-height:500px;object-fit:contain}</style>'
            page+='<h1>'+title+'</h1><p>'+html.escape(self.policy['name'])+' · 256 × 256 · sigma 4 · '+html.escape(self.policy['scoring'])+'</p>'
            page+=f'<p>图像分数 {score:.8f} / 图像阈值 {it:.10f}<br>像素阈值 {pt:.10f}；候选区域 {int(mask.sum()):,} 像素</p>'
            if result['experimental_candidate']:page+='<p><b>实验候选：</b>'+html.escape(result['evidence_limit'])+'</p>'
            page+='<p>'+html.escape(result['note'])+'</p><div class="grid">'+figures+'</div><p><a href="结果.json">版本和结果</a> · <a href="连续异常分数.npz">原始连续分数</a></p></html>'
            (out/'检测报告.html').write_text(page,encoding='utf-8')
            save_json(out/'status.json',{'status':'completed'})
            return out,result
        except Exception:
            save_json(out/'status.json',{'status':'failed'});raise
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',choices=['r18_reference','wrn50_a','wrn50_c_experimental'],default='r18_reference')
    ap.add_argument('--device',choices=['auto','cpu','cuda:0'],default='auto')
    ap.add_argument('--image',type=Path);ap.add_argument('--input-dir',type=Path);ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--check',action='store_true',help='verify checkpoint SHA/config; no inference')
    args=ap.parse_args()
    if args.image and args.input_dir:ap.error('--image与--input-dir互斥')
    engine=DeliveryInspector(args.profile,args.device,args.output_dir)
    if args.check:
        print(json.dumps(dict(profile=args.profile,policy=engine.policy,model_loaded=False),ensure_ascii=False));return
    if not args.image and not args.input_dir:ap.error('请指定--image或--input-dir，或使用--check')
    if args.input_dir and not args.input_dir.is_dir():ap.error('批量输入目录不存在')
    paths=[args.image] if args.image else sorted(f for f in args.input_dir.iterdir() if f.is_file() and f.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'})
    if not paths:ap.error('没有可支持的图片')
    items=[];failed=0
    for f in paths:
        try:
            out,r=engine.infer(f);items.append(dict(input=str(f),status='completed',output=str(out),score=r['raw_image_score'],anomalous=r['image_is_anomalous']))
        except Exception as e:
            failed+=1;items.append(dict(input=str(f),status='rejected',error=str(e)))
            logging.error('%s: %s',f,e)
    engine.output_base.mkdir(parents=True,exist_ok=True)
    report=engine.output_base/datetime.now().strftime('batch_%Y%m%d_%H%M%S_%f.json')
    save_json(report,dict(profile=args.profile,items=items,failures=failed,training_performed=False))
    print(json.dumps(dict(batch_report=str(report),completed=len(items)-failed,failed=failed),ensure_ascii=False))
    if failed:raise SystemExit(2)
if __name__=='__main__':main()
