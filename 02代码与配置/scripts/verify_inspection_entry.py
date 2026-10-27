"""Integration checks: three allowlisted development examples; no held-out files."""
from pathlib import Path
import sys
import time
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'02代码与配置/scripts'))
from inspection_runtime import Inspector, DATA, save_json
engine=Inspector()
source=engine.checkpoint.parent
old={r['image_path']:r for r in json.loads((source/'development_predictions.json').read_text(encoding='utf-8'))}
paths=['metal_nut/test/scratch/000.png','metal_nut/test/color/001.png','metal_nut/test/good/005.png']
checks=[]
for path in paths:
    assert path in engine.guard.manifest['splits']['development']
    output,r=engine.infer(DATA/path)
    with np.load(source/old[path]['map_path'],allow_pickle=False) as z: expected=z['anomaly_map']
    with np.load(output/'连续异常分数.npz',allow_pickle=False) as z: observed=z['anomaly_map']
    difference=float(np.abs(expected-observed).max())
    delta=abs(r['raw_image_score']-old[path]['raw_score'])
    threshold=r['pixel_threshold']
    changed=int(np.count_nonzero((observed>threshold)!=(expected>threshold)))
    row={'image_path':path,'max_absolute_map_difference':difference,'image_score_difference':delta,
         'pixel_decision_changes':changed,'image_decision_matches':r['image_is_anomalous']==(old[path]['raw_score']>r['image_threshold']),
         'output':output.relative_to(ROOT).as_posix()}
    checks.append(row);print(row,flush=True)
    assert np.allclose(expected,observed,atol=1e-5,rtol=1e-5), row
    assert delta<1e-5 and changed==0 and row['image_decision_matches'],row
report=ROOT/'04实验结果/14单图检测/入口验证'
report.mkdir(parents=True,exist_ok=True)
stamp=time.strftime('%Y%m%d_%H%M%S')
save_json(report/('verification_'+stamp+'.json'),{'status':'passed','development_examples':checks,'held_out_images_opened':0,'training_runs':0,'scope':'three-example implementation consistency check; no new performance benchmark'})
print('VERIFICATION_PASSED',flush=True)
