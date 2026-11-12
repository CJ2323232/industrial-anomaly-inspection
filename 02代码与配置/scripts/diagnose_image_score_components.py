"""Development-only decomposition of frozen PatchCore distance, weight and memory provenance."""
import csv,gc,hashlib,json,os,types
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for k,r in {'TORCH_HOME':'.cache/models/torch','HF_HOME':'.cache/models/huggingface','XDG_CACHE_HOME':'.cache/models','TEMP':'.cache/tmp','TMP':'.cache/tmp'}.items():os.environ[k]=str(ROOT/r)
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from run_trusted_baseline import load_checkpoint,make_transform,load_tensor
from baseline_utils import file_sha256,load_manifest
from sklearn.metrics import roc_auc_score,average_precision_score
B=ROOT/'04实验结果/09骨干网络对照/backbone_comparison_20261025_220104_486177'
R=ROOT/'04实验结果/01模型与检索实验/baseline_20261016_204723_088238'
DATA=ROOT/'03原始数据/解压数据/MVTecAD'
CASES=['metal_nut/test/good/005.png','metal_nut/test/good/006.png','metal_nut/test/scratch/021.png','metal_nut/test/scratch/000.png','metal_nut/test/scratch/001.png','metal_nut/test/scratch/003.png']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def embed(model,tensor):
    raw=model.feature_extractor(tensor)
    return model.reshape_embedding(model.generate_embedding({k:model.feature_pooler(v) for k,v in raw.items()}))
def main():
    out=ROOT/'04实验结果/18开发集评分分解'/datetime.now().strftime('score_diagnosis_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False);os.environ['MPLCONFIGDIR']=str(out/'.matplotlib')
    manifest=load_manifest(ROOT/'02代码与配置/configs/splits/metal_nut-v1.json')
    assert not set(CASES)-set(manifest['splits']['development'])
    provenance=read(B/'wide_resnet50/coreset_indices.json')
    assert provenance['fit_paths_in_order']==manifest['splits']['fit']
    records={'resnet18':csvrows(R/'normal_calibration_predictions.csv')+csvrows(R/'development_predictions.csv'),
             'wide_resnet50':read(B/'wide_resnet50/calibration_predictions.json')+read(B/'wide_resnet50/development_predictions.json')}
    checkpoints={'resnet18':R/'model.pt','wide_resnet50':B/'wide_resnet50/model.pt'}
    fixed=ROOT/'02代码与配置/configs/decisions/WRN50_v2.json'
    hashes={p.relative_to(ROOT).as_posix():file_sha256(p) for p in [fixed,ROOT/'02代码与配置/configs/decisions/metal_nut-wrn50-selected-v1.json',*checkpoints.values(),B/'wide_resnet50/coreset_indices.json']}
    frozen=read(fixed)
    for p,h in frozen['file_sha256'].items():assert file_sha256(ROOT/p)==h
    write(out/'protocol.json',dict(purpose='Frozen model scoring diagnosis only',started_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),splits=['normal_calibration','development'],held_out_images_opened=0,training_runs=0,threshold_changes=0,model_forwards_expected=178,normal_source_feature_checks='Additional forward feature extraction on fit normals only; no bank rebuilding',cases=CASES,threshold_selection=False,raw_distance_vs_weight_comparison='Ranking only; no threshold sweep',input_sha256=hashes))
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    allrows={};stats={};vectorchecks=[];fit_forwards=0
    for name,rs in records.items():
        model,saved=load_checkpoint(checkpoints[name],device);model.requires_grad_(False)
        transform=make_transform(saved['model_config']);capture={};original=model.compute_anomaly_score
        def wrapped(self,patch_scores,locations,embedding):
            reference=original(patch_scores,locations,embedding)
            imax=int(patch_scores[0].argmax());dmax=patch_scores[0,imax]
            bankindex=int(locations[0,imax]);q=embedding[imax:imax+1]
            _,neighbors=self.nearest_neighbors(self.memory_bank[bankindex:bankindex+1],n_neighbors=9)
            distances=self.euclidean_dist(q.unsqueeze(1),self.memory_bank[neighbors]).squeeze(1)
            prob=F.softmax(distances,dim=1);weight=1-prob[0,0]
            torch.testing.assert_close(weight*dmax,reference[0],rtol=1e-6,atol=1e-7)
            capture.clear();capture.update(max_patch_index=imax,max_patch_grid_rc=[imax//32,imax%32],dmax=float(dmax),weight=float(weight),image_score=float(reference[0]),nearest_bank_row=bankindex,support_bank_rows=neighbors[0].cpu().tolist(),support_query_distances=distances[0].cpu().tolist(),support_softmax=prob[0].cpu().tolist(),patch_distances=patch_scores[0].cpu().numpy().copy(),bank_locations=locations[0].cpu().numpy().copy())
            return reference
        model.compute_anomaly_score=types.MethodType(wrapped,model)
        rows=[]
        for i,old in enumerate(rs):
            path=old['image_path'];rec=manifest['files'][path]
            split='development' if path in manifest['splits']['development'] else 'normal_calibration'
            assert path in manifest['splits'][split] and path not in manifest['splits']['held_out']
            assert file_sha256(DATA/path)==rec['image_sha256']
            with torch.inference_mode():
                tensor=load_tensor(DATA,path,transform).unsqueeze(0).to(device);pred=model(tensor)
                score=float(pred.pred_score[0]);a=F.interpolate(pred.anomaly_map,size=(rec['height'],rec['width']),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            assert np.isclose(score,float(old['raw_score']),rtol=1e-6,atol=1e-6)
            with np.load((R if name=='resnet18' else B/'wide_resnet50')/old['map_path'],allow_pickle=False) as z:maperror=float(np.abs(a-z['anomaly_map']).max())
            assert maperror<1e-5
            raw=capture.pop('patch_distances');locations=capture.pop('bank_locations')
            dest=out/name/'patch_scores'/Path(path).with_suffix('.npz');dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,patch_distances=raw.reshape(32,32),nearest_bank_rows=locations.reshape(32,32))
            if rec['mask_path']:
                assert file_sha256(DATA/rec['mask_path'])==rec['mask_sha256']
                gt=np.asarray(Image.open(DATA/rec['mask_path']))>0
            else:gt=np.zeros((rec['height'],rec['width']),dtype=bool)
            grid=F.interpolate(torch.from_numpy(gt.astype(np.float32))[None,None],size=(32,32),mode='area')[0,0].numpy()
            rr=dict(image_path=path,split=split,label=rec['label'],**capture,old_image_score=float(old['raw_score']),saved_score_abs_difference=abs(score-float(old['raw_score'])),saved_map_max_abs_difference=maperror,weight_reduction=float(capture['dmax']-score),patch_scores=dest.relative_to(out).as_posix(),max_cell_GT_fraction=float(grid.ravel()[capture['max_patch_index']]),max_GT_intersecting_cell_distance=float(raw[grid.ravel()>0].max()) if gt.any() else None,nominal_grid_cell_not_receptive_field=True)
            if name=='wide_resnet50':
                support=[]
                for j in rr['support_bank_rows']:
                    poolindex=int(provenance['indices'][j]);imidx,cell=divmod(poolindex,1024)
                    support.append(dict(bank_row=j,candidate_pool_index=poolindex,source_image=provenance['fit_paths_in_order'][imidx],source_grid_rc=[cell//32,cell%32]))
                rr['support_provenance']=support
                rr['normal_nearest_source']=next(s for s in support if s['bank_row']==rr['nearest_bank_row'])
            rows.append(rr)
            if (i+1)%15==0:print(name,i+1,'/89',flush=True)
        model.compute_anomaly_score=original
        # Verify WRN coreset source identity for the selected diagnostic cases.
        if name=='wide_resnet50':
            wanted={}
            for r in rows:
                if r['image_path'] in CASES:
                    for p in r['support_provenance']:wanted.setdefault(p['source_image'],{})[p['bank_row']]=p
            with torch.inference_mode():
                for path,entries in wanted.items():
                    assert path in manifest['splits']['fit'] and manifest['files'][path]['label']==0
                    assert file_sha256(DATA/path)==manifest['files'][path]['image_sha256']
                    features=embed(model,load_tensor(DATA,path,transform).unsqueeze(0).to(device));fit_forwards+=1
                    for j,p in entries.items():
                        cell=p['source_grid_rc'][0]*32+p['source_grid_rc'][1]
                        err=float(torch.abs(features[cell]-model.memory_bank[j]).max())
                        assert err<1e-5
                        vectorchecks.append(dict(**p,max_abs_vector_difference=err))
            print('normal source vectors verified',len(vectorchecks),'from fit images',fit_forwards,flush=True)
        allrows[name]=rows;write(out/(name+'_decomposition.json'),rows)
        dev=[r for r in rows if r['split']=='development'];cal=[r for r in rows if r['split']=='normal_calibration']
        stats[name]={}
        for group,rr in [('normal_calibration',cal),('development_normal',[r for r in dev if not r['label']]),('development_anomaly',[r for r in dev if r['label']])]:
            stats[name][group]=dict(n=len(rr),dmax_range=[min(r['dmax'] for r in rr),max(r['dmax'] for r in rr)],weight_range=[min(r['weight'] for r in rr),max(r['weight'] for r in rr)],weight_mean=float(np.mean([r['weight'] for r in rr])))
        y=[r['label'] for r in dev]
        stats[name]['development_ranking']={k:dict(AUROC=float(roc_auc_score(y,[r[k] for r in dev])),AP=float(average_precision_score(y,[r[k] for r in dev]))) for k in ['dmax','image_score']}
        neg=[r for r in dev if not r['label']];pos=[r for r in dev if r['label']]
        stats[name]['normal_anomaly_pair_order_flips_from_weight']=sum((n['dmax']<p['dmax'])!=(n['image_score']<p['image_score']) for n in neg for p in pos)
        del model,saved;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    write(out/'summary.json',dict(groups=stats,WRN_normal_source_vector_checks=vectorchecks,normal_source_fit_image_forwards=fit_forwards))
    assert all(file_sha256(ROOT/p)==h for p,h in hashes.items())
    for p,h in frozen['file_sha256'].items():assert file_sha256(ROOT/p)==h
    write(out/'verification.json',dict(frozen_files_unchanged=True,input_sha256=hashes,script_sha256=file_sha256(Path(__file__)),existing_scores_reproduced=True,existing_maps_reproduced=True,weight_times_distance_equals_reference=True,held_out_images_opened=0,scored_image_forwards=178,normal_fit_feature_forwards=fit_forwards,normal_source_vectors_verified=len(vectorchecks),threshold_changes=0,training_runs=0))
    write(out/'status.json',dict(status='completed',completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),held_out_images_opened=0))
    print('COMPLETED',str(out),flush=True);print(json.dumps(stats,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
