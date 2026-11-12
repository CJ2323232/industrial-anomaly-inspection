"""Read-only-input scoring diagnostic; outputs only in this new experiment directory."""
import os, sys, json, math, hashlib, platform
from pathlib import Path
from datetime import datetime
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(OUT / "plot_cache")
import numpy as np
from PIL import Image
from scipy.ndimage import label as component_labels
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_json(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))
def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def relative(p): return p.relative_to(ROOT).as_posix()
protocol=read_json(OUT/"protocol.json")
assert protocol["K_rule"]["K"] == 11 and protocol["calibration"]["q"] == .975
K=11
D=ROOT/protocol["source"]
BASE=ROOT/"04实验结果/09骨干网络对照/backbone_comparison_20261025_220104_486177/wide_resnet50"
manifest_path=ROOT/"02代码与配置/configs/splits/metal_nut-v1.json"
freeze_path=ROOT/"02代码与配置/configs/decisions/WRN50_v2.json"
manifest=read_json(manifest_path)
freeze=read_json(freeze_path)
source_path=D/"wide_resnet50_decomposition.json"
source=read_json(source_path)
allowed={s:set(manifest["splits"][s]) for s in ["normal_calibration","development"]}
assert len(source)==89 and len({r["image_path"] for r in source})==89
assert {r["image_path"] for r in source}==allowed["normal_calibration"]|allowed["development"]
assert not ({r["image_path"] for r in source}&set(manifest["splits"]["held_out"]))
data_root=ROOT/"03原始数据/解压数据/MVTecAD"
inputs=[OUT/"protocol.json",source_path,D/"protocol.json",D/"verification.json",
        manifest_path,freeze_path,BASE/"config.json",BASE/"calibration.json",
        BASE/"model.pt",BASE/"coreset_indices.json",
        ROOT/"02代码与配置/scripts/diagnose_image_score_components.py",
        ROOT/"02代码与配置/scripts/run_trusted_baseline.py",
        ROOT/".venv/Lib/site-packages/anomalib/models/image/patchcore/torch_model.py",
        ROOT/".venv/Lib/site-packages/anomalib/models/image/patchcore/anomaly_map.py",
        ROOT/".venv/Lib/site-packages/anomalib/models/components/filters/blur.py"]
for r in source:
    assert r["image_path"] in allowed[r["split"]]
    assert r["label"]==manifest["files"][r["image_path"]]["label"]
    inputs.append(D/r["patch_scores"])
    if r["split"]=="development" and r["label"]:
        inputs.append(data_root/manifest["files"][r["image_path"]]["mask_path"])
before={relative(p):digest(p) for p in inputs}
assert before[relative(BASE/"model.pt")]==freeze["checkpoint_sha256"]
assert freeze["pixel_threshold"]==protocol["pixel_threshold"]
save(OUT/"input_hashes_before.json",before)
save(OUT/"execution_start.json",{
    "started_at":datetime.now().astimezone().isoformat(),
    "locked_protocol_sha256":before[relative(OUT/"protocol.json")],
    "script_sha256":digest(Path(__file__)),
    "python":platform.python_version(),"numpy":np.__version__,
    "inference":False,"training":False,"allowed_images":sorted(set.union(*allowed.values())),
    "heldout_data_policy":"Manifest membership only; no images, GT, scores or predictions opened."
})
rows=[]; arrays={}; gt_grids={}
# Float32 source distances are retained exactly; C reduction uses float64 arithmetic.
for r in source:
    with np.load(D/r["patch_scores"], allow_pickle=False) as z:
        grid=z["patch_distances"].copy()
    assert grid.shape==(32,32) and grid.dtype==np.float32 and np.isfinite(grid).all()
    flat=grid.astype(np.float64).ravel()
    # Descending value; ties select smaller row-major index first.
    top=np.lexsort((np.arange(1024),-flat))[:K]
    a=float(r["image_score"]); b=float(flat.max()); c=float(flat[top].mean())
    assert b==r["dmax"]
    assert abs(float(np.float32(r["weight"])*np.float32(b))-a)<2e-6
    assert r["saved_score_abs_difference"]==0 and r["saved_map_max_abs_difference"]==0
    # Upper-tail support contains ties at the eleventh largest value.
    cutoff=float(flat[top[-1]])
    support=grid>=cutoff
    cc,ncc=component_labels(support,structure=np.ones((3,3),dtype=int))
    max_index=int(np.argmax(flat)); max_rc=np.unravel_index(max_index,grid.shape)
    peak_id=cc[max_rc]
    gt=np.zeros((32,32),dtype=float)
    area=None
    if r["split"]=="development":
        area=0.0
        if r["label"]:
            meta=manifest["files"][r["image_path"]]
            mask_path=data_root/meta["mask_path"]
            assert digest(mask_path)==meta["mask_sha256"]
            with Image.open(mask_path) as im:
                mask=np.asarray(im)>0
            assert mask.shape==(meta["height"],meta["width"])
            area=float(mask.mean())
            # Area/adaptive-average cells, same floor/ceil convention as area interpolation.
            for y in range(32):
                for x in range(32):
                    block=mask[math.floor(y*mask.shape[0]/32):math.ceil((y+1)*mask.shape[0]/32),
                               math.floor(x*mask.shape[1]/32):math.ceil((x+1)*mask.shape[1]/32)]
                    gt[y,x]=block.mean()
        gt_grids[r["image_path"]]=gt
    row={
        "image_path":r["image_path"],"split":r["split"],"label":r["label"],
        "scores":{"A":a,"B":b,"C":c},"neighbor_weight":r["weight"],
        "B_minus_C":b-c,"C_div_B":c/b,
        "topK_indices":top.tolist(),"topK_cutoff":cutoff,"topK_count":K,
        "upper_tail_count_including_ties":int(support.sum()),
        "upper_tail_8connected_components":int(ncc),
        "max_cell_upper_tail_component_size":int((cc==peak_id).sum()),
        "max_cell_is_isolated_in_upper_tail":bool((cc==peak_id).sum()==1),
        "max_patch_index":max_index,
        "gt_area_fraction":area,
        "small_defect_lt_1pct":bool(r["split"]=="development" and r["label"] and area<.01),
        "topK_GT_intersecting_cells":int((gt.ravel()[top]>0).sum()) if r["split"]=="development" else None,
        "topK_mean_GT_cell_fraction":float(gt.ravel()[top].mean()) if r["split"]=="development" else None,
        "GT_cell_measure":"adaptive area average on nominal 32x32 grid; not a receptive-field mask",
        "source_patch_file":relative(D/r["patch_scores"])
    }
    rows.append(row); arrays[r["image_path"]]=grid
cal=[r for r in rows if r["split"]=="normal_calibration"]
dev=[r for r in rows if r["split"]=="development"]
normal=[r for r in dev if not r["label"]]; abnormal=[r for r in dev if r["label"]]
assert (len(cal),len(dev),len(normal),len(abnormal))==(44,45,8,37)
metrics={}; thresholds={}; quantiles={}; dist={}
names={"A":"原始加权最大距离","B":"未加权最大距离","C":"Top-11 平均距离"}
def describe(values):
    v=np.asarray(values,dtype=float)
    return dict(n=len(v),mean=float(v.mean()),std_population=float(v.std()),
                min=float(v.min()),q25=float(np.quantile(v,.25)),median=float(np.median(v)),
                q75=float(np.quantile(v,.75)),max=float(v.max()))
for key in ["A","B","C"]:
    sorted_cal=sorted(cal,key=lambda r:(r["scores"][key],r["image_path"]))
    tau=float(np.quantile([r["scores"][key] for r in cal],.975,method="linear"))
    thresholds[key]=tau
    index=(len(cal)-1)*.975
    lo=math.floor(index); hi=math.ceil(index); frac=index-lo
    manual=(1-frac)*sorted_cal[lo]["scores"][key]+frac*sorted_cal[hi]["scores"][key]
    assert abs(manual-tau)<1e-14
    quantiles[key]=dict(q=.975,method="linear",n=44,index_zero_based=index,
                       low_rank_one_based=lo+1,high_rank_one_based=hi+1,
                       low_image=sorted_cal[lo]["image_path"],low_score=sorted_cal[lo]["scores"][key],
                       high_image=sorted_cal[hi]["image_path"],high_score=sorted_cal[hi]["scores"][key],
                       low_weight=1-frac,high_weight=frac,threshold=tau)
    for r in rows:
        r.setdefault("predictions",{})[key]=int(r["scores"][key]>tau)
        r.setdefault("margins",{})[key]=r["scores"][key]-tau
    tp=sum(r["predictions"][key] for r in abnormal)
    fp=sum(r["predictions"][key] for r in normal)
    scores=[r["scores"][key] for r in dev]; labels=[r["label"] for r in dev]
    auc=float(roc_auc_score(labels,scores)); ap=float(average_precision_score(labels,scores))
    cross=[np.sign(a["scores"][key]-n["scores"][key]) for a in abnormal for n in normal]
    wins=int(sum(v>0 for v in cross)); ties=int(sum(v==0 for v in cross)); losses=int(sum(v<0 for v in cross))
    assert abs(auc-(wins+.5*ties)/296)<1e-12
    metrics[key]=dict(AUROC=auc,AP=ap,TP=tp,FN=37-tp,FP=fp,TN=8-fp,
                      Recall=tp/37,FPR=fp/8,Precision=tp/(tp+fp) if tp+fp else None,
                      calibration_exceedances=sum(r["predictions"][key] for r in cal),
                      correct_crossclass_pairs=wins,tied_crossclass_pairs=ties,inverted_crossclass_pairs=losses,
                      normal_max=max(r["scores"][key] for r in normal),
                      anomalous_min=min(r["scores"][key] for r in abnormal),
                      raw_class_gap=min(r["scores"][key] for r in abnormal)-max(r["scores"][key] for r in normal),
                      FP_images=[r["image_path"] for r in normal if r["predictions"][key]],
                      FN_images=[r["image_path"] for r in abnormal if not r["predictions"][key]])
    dist[key]={name:describe([r["scores"][key] for r in subset])
               for name,subset in [("normal_calibration",cal),("development_normal",normal),("development_anomalous",abnormal)]}
    save(OUT/key/"config.json",dict(group=key,name=names[key],formula=protocol["groups"][key],
          fixed_model=freeze["checkpoint"],checkpoint_sha256=freeze["checkpoint_sha256"],
          fixed_preprocessing=freeze["preprocessing"],fixed_pixel_threshold=freeze["pixel_threshold"],
          runtime_image_threshold_unchanged=freeze["image_threshold"],
          analysis_image_threshold=tau,analysis_calibration=protocol["calibration"],
          patch_count=1024,K=11 if key=="C" else None,
          accumulation="float64 mean of saved float32 distances" if key=="C" else "saved reference float32",
          config_scope="Diagnostic artifact only, not a runtime configuration",
          model_forwards=0,training_runs=0))
    save(OUT/key/"scores.json",[dict(image_path=r["image_path"],split=r["split"],label=r["label"],
         raw_score=r["scores"][key],threshold=tau,prediction=r["predictions"][key],
         margin=r["margins"][key]) for r in rows])
    save(OUT/key/"metrics.json",metrics[key])
pairs=[]
for a in abnormal:
    for n in normal:
        relations={k:int(np.sign(a["scores"][k]-n["scores"][k])) for k in names}
        pairs.append(dict(anomalous_image=a["image_path"],normal_image=n["image_path"],relations=relations))
changes={}
for first,second in [("A","B"),("A","C"),("B","C")]:
    key=first+"_to_"+second
    changes[key]={
        "pair_relation_changes":sum(r["relations"][first]!=r["relations"][second] for r in pairs),
        "strict_reversals":sum(r["relations"][first]*r["relations"][second]==-1 for r in pairs),
        "prediction_changes":[dict(image_path=r["image_path"],label=r["label"],
           before=r["predictions"][first],after=r["predictions"][second]) for r in dev
           if r["predictions"][first]!=r["predictions"][second]]}
requested=["metal_nut/test/"+s+".png" for s in
           ["good/005","scratch/021","color/004","good/016","scratch/020","scratch/022"]]
requested_cases=[]
by_path={r["image_path"]:r for r in rows}
for p in requested:
    split=next(s for s,v in manifest["splits"].items() if p in v)
    requested_cases.append(by_path[p] if split=="development" else
        dict(image_path=p,split=split,scores=None,predictions=None,
             reason="N/A: belongs to held_out; no image, GT, score or prediction loaded in this experiment",
             known_miss_named_in_request=p in [requested[2],requested[4],requested[5]]))
attenuation={}
for key,subset in [("normal_calibration",cal),("development_normal",normal),("development_anomalous",abnormal),
                   ("small_defect_lt_1pct",[r for r in abnormal if r["small_defect_lt_1pct"]])]:
    attenuation[key]=dict(absolute_drop=describe([r["B_minus_C"] for r in subset]),
                         retention_ratio=describe([r["C_div_B"] for r in subset]),
                         isolated_max_count=sum(r["max_cell_is_isolated_in_upper_tail"] for r in subset))
summary=dict(group_names=names,thresholds=thresholds,metrics=metrics,distributions=dist,
             quantile_derivation=quantiles,comparisons=changes,attenuation=attenuation,
             sample_counts=dict(normal_calibration=44,development=45,normal=8,anomalous=37),
             fixed_pixel_threshold=freeze["pixel_threshold"],model_forwards=0,training_runs=0,
             heldout_data_reads=0,classification_rule="strict score > own calibration threshold",
             K_rule=protocol["K_rule"],calibration_history="q=.975 historically development-informed; diagnostic only")
save(OUT/"summary.json",summary)
save(OUT/"paired_scores.json",rows)
save(OUT/"crossclass_pairs.json",pairs)
save(OUT/"requested_cases.json",requested_cases)
save(OUT/"small_defect_cases.json",[r for r in abnormal if r["small_defect_lt_1pct"]])
save(OUT/"allowed_data_read_log.json",{
    "patch_distance_npz_files":[relative(D/r["patch_scores"]) for r in source],
    "development_GT_files":[relative(data_root/manifest["files"][r["image_path"]]["mask_path"]) for r in source if r["split"]=="development" and r["label"]],
    "source_decomposition_json":relative(source_path),"manifest_membership_only_heldout_cases":requested_cases[2:],
    "raw_image_files_opened":0,"heldout_images_GT_predictions_opened":0,
    "note":"Recorded by this analysis script; supporting model/config/source files are listed in input hashes."
})
plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"figure.dpi":120})
fig,axes=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
for ax,key in zip(axes,names):
    for level,(label_,subset,color) in enumerate([("正常校准 44",cal,"#2879b8"),("开发正常 8",normal,"#2a9d60"),("开发异常 37",abnormal,"#e78525")]):
        vals=[r["scores"][key] for r in subset]
        jitter=np.linspace(-.13,.13,len(vals))
        ax.scatter(vals,level+jitter,s=24,c=color,alpha=.8,label=label_)
    ax.axvline(thresholds[key],c="#bc2945",ls="--",label="本组正常校准 q97.5%")
    ax.set_yticks([0,1,2],["正常校准 44","开发正常 8","开发异常 37"])
    ax.set_title(key+"："+names[key]+f"；阈值 {thresholds[key]:.6f}",loc="left")
    ax.set_xlabel("本组原始图像分数（非概率；各面板尺度不同）")
    ax.grid(axis="x",alpha=.2)
axes[0].legend(loc="upper left",fontsize=8,ncol=2)
fig.savefig(OUT/"三组分数分布.png");plt.close(fig)
fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True)
for key,style in zip(names,["-","--",":"]):
    fpr,tpr,_=roc_curve([r["label"] for r in dev],[r["scores"][key] for r in dev])
    ax.plot(fpr,tpr,style,lw=2,label=f'{key}：AUROC={metrics[key]["AUROC"]:.4f}')
    ax.scatter(metrics[key]["FPR"],metrics[key]["Recall"],s=75,marker={"A":"o","B":"x","C":"s"}[key])
ax.plot([0,1],[0,1],c="gray",alpha=.4)
ax.set(xlabel="图像误报率 FPR",ylabel="图像召回率 Recall",title="开发集 ROC 与正常校准工作点（不是阈值搜索）",xlim=(-.03,1.03),ylim=(-.03,1.03))
ax.legend();ax.grid(alpha=.2)
fig.savefig(OUT/"ROC与固定规则工作点.png");plt.close(fig)
case_paths=list(dict.fromkeys(requested[:2]+[r["image_path"] for r in abnormal if r["small_defect_lt_1pct"]]))
fig,axes=plt.subplots(len(case_paths),2,figsize=(10,3.4*len(case_paths)),squeeze=False,constrained_layout=True)
for (ax1,ax2),path in zip(axes,case_paths):
    r=by_path[path];g=arrays[path];gt=gt_grids[path]
    im=ax1.imshow(g,cmap="viridis");fig.colorbar(im,ax=ax1,shrink=.8,label="未加权原始距离")
    top=np.array(r["topK_indices"]); yy,xx=np.divmod(top,32)
    ax1.scatter(xx,yy,facecolors="none",edgecolors="white",s=50,linewidths=1)
    if gt.max()>0: ax1.contour(gt,levels=[.001],colors=["#ff3b3b"],linewidths=1)
    ax1.set_title(path.replace("metal_nut/test/","")+f'\n白圈=Top11；红线=GT粗网格；峰值分量={r["max_cell_upper_tail_component_size"]}格')
    ax1.set_xticks([]);ax1.set_yticks([])
    ax2.plot(range(1,12),np.sort(g.ravel())[-11:][::-1],"o-",label="前11个距离")
    ax2.axhline(r["scores"]["C"],c="#cc7733",ls="--",label="Top11均值")
    ax2.set(xlabel="Patch 排名",ylabel="原始距离",
            title=f'B−C={r["B_minus_C"]:.4f}；保留 {100*r["C_div_B"]:.2f}%\nTop11中与GT粗网格相交={r["topK_GT_intersecting_cells"]}/11')
    ax2.legend(fontsize=8);ax2.grid(alpha=.2)
fig.suptitle("固定K的空间与尾部分数诊断；粗网格不是独立裁块或精确感受野")
fig.savefig(OUT/"Top11空间诊断.png");plt.close(fig)
after={relative(p):digest(p) for p in inputs}
assert before==after
save(OUT/"verification.json",{
    "all_input_hashes_unchanged":True,"input_file_count":len(inputs),"hashes_after":after,
    "checkpoint_matches_frozen_sha256":True,"A_reference_score_difference":0.0,
    "B_matches_saved_dmax_exactly":True,"C_sorted_top11_mean_float64":True,
    "AUROC_matches_all_296_pair_relations":True,"quantile_matches_manual_interpolation":True,
    "pixel_threshold_unchanged":True,"pixel_map_recomputed":False,"pixel_mask_changed_by_this_analysis":False,
    "model_forward_calls":0,"training_runs":0,"heldout_image_GT_prediction_reads":0,
    "script_sha256":digest(Path(__file__)),"finished_at":datetime.now().astimezone().isoformat()
})
print(json.dumps(dict(metrics=metrics,thresholds=thresholds,comparisons=changes,
                     small_defects=[dict(path=r["image_path"],area=r["gt_area_fraction"],scores=r["scores"],pred=r["predictions"],ratio=r["C_div_B"],topGT=r["topK_GT_intersecting_cells"]) for r in abnormal if r["small_defect_lt_1pct"]],
                     attenuation=attenuation),ensure_ascii=False,indent=2))
