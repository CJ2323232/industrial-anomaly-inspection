"""Saved-score image threshold analysis. No held-out file access or inference."""
import csv, hashlib, json, os
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'04实验结果/09骨干网络对照/backbone_comparison_20261025_220104_486177'
R=ROOT/'04实验结果/01模型与检索实验/baseline_20261016_204723_088238'
D=ROOT/'02代码与配置/configs/decisions'
CANDIDATE=D/'metal_nut-wrn50-image-v2-candidate.json'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def counts(labels,scores,threshold):
    y=np.asarray(labels,dtype=bool); p=np.asarray(scores)>threshold
    tp,fp,fn,tn=[int(z.sum()) for z in [y&p,~y&p,y&~p,~y&~p]]
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,Recall=tp/(tp+fn),FPR=fp/(fp+tn))
def select_zero_miss(labels,scores):
    """Require FN=0, minimize FP, then maximize distance to nearest observed scores.
    Strict score>tau; tied scores cannot be separated. Uses development labels.
    """
    y=np.asarray(labels,dtype=bool); x=np.asarray(scores,dtype=np.float64)
    if y.ndim!=1 or y.shape!=x.shape or not y.any() or y.all() or not np.isfinite(x).all():
        raise ValueError('Finite aligned scores and both classes are required')
    upper=float(x[y].min()); below=x[x<upper]; lower=float(below.max()) if below.size else None
    tau=(lower+upper)/2 if lower is not None else float(np.nextafter(upper,-np.inf))
    if tau>=upper: tau=float(np.nextafter(upper,-np.inf))
    c=counts(y,x,tau); assert c['FN']==0
    return dict(threshold=tau,lower_score=lower,upper_score=upper,minimum_observed_margin=float(np.abs(x-tau).min()),**c)
def main():
    if CANDIDATE.exists(): raise FileExistsError('Preserve existing candidate; create a new version explicitly.')
    out=ROOT/'04实验结果/16图像级阈值校准'/datetime.now().strftime('image_threshold_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False); os.environ['MPLCONFIGDIR']=str(out/'.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix
    mp=ROOT/'02代码与配置/configs/splits/metal_nut-v1.json'; oldp=D/'metal_nut-wrn50-selected-v1.json'
    sources=[mp,oldp,B/'report.json',R/'development_predictions.csv',R/'normal_calibration_predictions.csv',B/'wide_resnet50/development_predictions.json',B/'wide_resnet50/calibration_predictions.json',D/'metal_nut-wrn50-frozen-v1.json']
    sources += [ROOT/p for p in read(D/'metal_nut-wrn50-frozen-v1.json')['file_sha256']]
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    manifest,old=read(mp),read(oldp)
    rule='Development FN=0; minimize FP; maximize nearest-observation absolute score margin within the optimal plateau.'
    protocol=dict(inputs='Saved normal_calibration/development image raw_score only',selection_rule=rule,operator='score > threshold',quantile_grid=[.95,.975,.99,.995,1.0],cross_validation='20 repeats, stratified 5 folds, seeds 0..19; select on four training folds only',test_data_accessed=False,post_test_development=True,training_runs=0,model_forwards=0,limitation='Previous v1 test errors motivated this work; no untouched-test claim for v2.')
    write(out/'protocol.json',protocol)
    def csvrows(p):
        with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
    dev={'resnet18':csvrows(R/'development_predictions.csv'),'wide_resnet50':read(B/'wide_resnet50/development_predictions.json')}
    cal={'resnet18':csvrows(R/'normal_calibration_predictions.csv'),'wide_resnet50':read(B/'wide_resnet50/calibration_predictions.json')}
    paths=sorted(manifest['splits']['development']); y=np.array([manifest['files'][p]['label'] for p in paths])
    scores,results,perimage,folds,scans={},{},{},{},{}
    for n,rr in dev.items():
        lookup={r['image_path']:r for r in rr}; cc=cal[n]
        assert len(rr)==len(lookup)==45 and set(lookup)==set(paths)
        assert len(cc)==44 and {r['image_path'] for r in cc}==set(manifest['splits']['normal_calibration'])
        assert all(int(r['label'])==0 and manifest['files'][r['image_path']]['label']==0 for r in cc)
        assert all(int(r['label'])==manifest['files'][r['image_path']]['label'] and r['image_sha256']==manifest['files'][r['image_path']]['image_sha256'] for r in rr+cc)
        scores[n]=x=np.array([float(lookup[p]['raw_score']) for p in paths])
        tau=float(np.quantile([float(r['raw_score']) for r in cc],.975,method='linear'));selected=select_zero_miss(y,x)
        grid=[]
        for q in protocol['quantile_grid']:
            t=float(np.quantile([float(r['raw_score']) for r in cc],q,method='linear'));grid.append(dict(q=q,threshold=t,**counts(y,x,t)))
        unique=np.unique(x);cutoffs=[float(np.nextafter(unique[0],-np.inf))]+[float((a+b)/2) for a,b in zip(unique[:-1],unique[1:])]+[float(unique[-1])]
        scans[n]=[dict(threshold=t,**counts(y,x,t)) for t in cutoffs]
        assert selected['FP']==min(p['FP'] for p in scans[n] if p['FN']==0)
        results[n]=dict(original=dict(threshold=tau,**counts(y,x,tau)),development_selected=selected,normal_only_quantiles=grid,normal_development_max=float(x[y==0].max()),anomalous_development_min=float(x[y==1].min()))
        perimage[n]=[dict(image_path=p,label=int(label),raw_score=float(s),old_prediction=int(s>tau),candidate_prediction=int(s>selected['threshold'])) for p,label,s in zip(paths,y,x)]
        oof=np.zeros((20,len(paths)),dtype=bool);folds[n]=[]
        for seed in range(20):
            for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(x,y)):
                choice=select_zero_miss(y[train],x[train]);t=choice['threshold'];assert set(train).isdisjoint(test)
                oof[seed,test]=x[test]>t
                folds[n].append(dict(seed=seed,fold=fold,threshold=t,train_paths=[paths[i] for i in train],validation_paths=[paths[i] for i in test],**counts(y[test],x[test],t)))
        for i,row in enumerate(perimage[n]): row['OOF_wrong_repeats_of_20']=int((oof[:,i]!=y[i]).sum())
        repeats=[]
        for seed,pred in enumerate(oof):
            tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();repeats.append(dict(seed=seed,TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn)))
        results[n]['cross_validation']=dict(repeats=repeats,mean_FP_per_repeat=float(np.mean([r['FP'] for r in repeats])),mean_FN_per_repeat=float(np.mean([r['FN'] for r in repeats])),threshold_range=[min(r['threshold'] for r in folds[n]),max(r['threshold'] for r in folds[n])],independent_images=45,independent_normals=8,independent_anomalies=37,note='Repeated predictions of the same images are dependent, not 900 independent samples.')
        tn,fp,fn,tp=confusion_matrix(y,x>selected['threshold'],labels=[0,1]).ravel()
        assert [int(tp),int(fp),int(fn),int(tn)]==[selected[k] for k in ['TP','FP','FN','TN']]
    candidate_tau=results['wide_resnet50']['development_selected']['threshold']
    assert old['image_threshold']==results['wide_resnet50']['original']['threshold']
    candidate=dict(old)
    candidate.update(version='metal_nut-wrn50-image-v2-candidate',status='supervised_development_image_threshold_candidate',recorded_on=datetime.now().isoformat(),image_threshold=candidate_tau,image_threshold_provenance='Development image labels: zero-FN minimum-FP maximal-margin plateau midpoint. Not normal-only calibration.',previous_image_threshold=old['image_threshold'],selection_rule=rule,evidence_directory=out.relative_to(ROOT).as_posix(),development_metrics=results['wide_resnet50']['development_selected'],cross_validation=results['wide_resnet50']['cross_validation'],source_policy=oldp.relative_to(ROOT).as_posix(),source_policy_sha256=sha(oldp),runtime_switch_performed=False,final_evaluation_authorized=False,held_out_images_opened=0,held_out_evaluation_for_this_version=False,post_test_development=True,limits='Uses development labels after v1 test diagnosis; needs new unseen validation. Same model and pixel output; not a proven universal upgrade.')
    write(out/'candidate_policy.json',candidate);write(CANDIDATE,candidate)
    write(out/'summary.json',dict(protocol=protocol,methods=results,candidate_policy=CANDIDATE.relative_to(ROOT).as_posix()))
    write(out/'per_image.json',perimage);write(out/'threshold_scan.json',scans);write(out/'cross_validation_folds.json',folds)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for n,color in [('resnet18','#4775b8'),('wide_resnet50','#e87b26')]:
        pts=scans[n];axes[0].plot([p['FPR'] for p in pts],[p['Recall'] for p in pts],'.-',label=n,color=color)
    axes[0].set(xlabel='Normal-image FPR (8 normals)',ylabel='Anomalous-image recall (37 anomalies)',title='Development operating points');axes[0].legend();axes[0].grid(alpha=.2)
    x=scores['wide_resnet50'];axes[1].scatter(x[y==0],np.zeros(8),label='Normal development (8)');axes[1].scatter(x[y==1],np.ones(37),label='Anomalous development (37)')
    axes[1].axvline(old['image_threshold'],ls='--',color='red',label='v1 threshold');axes[1].axvline(candidate_tau,ls='--',color='green',label='v2 supervised candidate')
    axes[1].set(xlabel='Raw image score (not probability)',yticks=[0,1],yticklabels=['Normal','Anomalous'],title='WRN50 saved image scores');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'图像级阈值对照.png',dpi=160);plt.close(fig)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    write(out/'verification.json',dict(existing_inputs_unchanged=True,input_sha256=hashes,script_sha256=sha(__file__),sklearn_confusion_checks_passed=True,test_files_read=0,model_forwards=0,all_threshold_plateaus_scanned=True))
    print(json.dumps(dict(output=str(out),candidate_threshold=candidate_tau,methods=results),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
