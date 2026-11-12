"""Label original sample membership and existing report figures without changing frozen artifacts."""
from pathlib import Path
from collections import Counter
import json
import html
import re
from baseline_utils import load_manifest
from inspection_runtime import ROOT, MANIFEST

LABEL={'fit':'正常建库集','normal_calibration':'正常校准集','development':'开发验证集','held_out':'保留独立测试集'}
manifest=load_manifest(MANIFEST)
membership={p:s for s,paths in manifest['splits'].items() for p in paths}
records=[{'image_path':p,'split':s,'split_label':LABEL[s],'label':manifest['files'][p]['label'],
          'defect_type':manifest['files'][p]['defect_type']} for p,s in membership.items()]
counts={s:Counter(manifest['files'][p]['defect_type'] for p in paths) for s,paths in manifest['splits'].items()}
out=ROOT/'05项目文档'
(out/'图片集合清单.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 文档里的图片属于哪个集合','',
'**以固定清单为准，不根据文件夹里有没有test这个词判断。** MVTec原始test目录中的115张图片，被本项目分成45张开发验证图和70张保留测试图。','',
'|集合|数量|正常/异常|在本项目中的用途|','|---|---:|---|---|',
f'|正常建库集|{len(manifest["splits"]["fit"])}|全正常|提取正常局部特征，建立PatchCore记忆库|',
f'|正常校准集|{len(manifest["splits"]["normal_calibration"])}|全正常|确定正常分数分布与原始阈值，不进入当前正常特征库|',
f'|开发验证集|{len(manifest["splits"]["development"])}|8正常＋37异常|比较网络、分辨率、分块、平滑和阈值，属于已经看过的数据|',
f'|保留独立测试集|{len(manifest["splits"]["held_out"])}|14正常＋56异常|模型及阈值冻结后本次正式评价，不能再用来调参|','',
'这里的“独立”指未用于本版本调参，不代表已经证明来自不同工厂、拍摄批次或物理对象。此前曾做文件完整性、尺寸与哈希审查；本次是首次正式模型性能评估，并非首次读取文件字节。','',
'骨干网络使用ImageNet预训练权重。上面的176张正常螺母图用于建库，不是重新预训练WideResNet50。早期试跑曾用全部220张正常图建库；当前冻结版本采用176/44分离方案，不能把早期模型当成当前模型。','',
'## 你已看过的图片','', '|图片路径|归属|','|---|---|']
for p in ['metal_nut/train/good/000.png','metal_nut/test/good/005.png','metal_nut/test/color/000.png','metal_nut/test/color/001.png','metal_nut/test/scratch/000.png','metal_nut/test/scratch/004.png','metal_nut/test/bent/014.png']:
    if p in membership:lines.append(f'|{p}|{LABEL[membership[p]]}|')
lines += ['', '## 按报告区分','',
'- 01可信基线的开发预测、02阈值分析中的开发部分、05分辨率、06平滑、07错误诊断、08分块、09骨干、10等召回、11阈值扫描、12定位失败、13阈值稳定性：用于方案比较的样例来自开发验证集。',
'- 02阈值分布图中，蓝色normal calibration的44张是正常校准集；绿色normal development的8张、橙色anomalous development的37张是开发验证集。它是一张混合集合的统计图。',
'- 13的5折验证：把同一批45张开发图分成5份，轮流留出一份；不会产生新的独立测试集。',
'- 14单图检测：内置三个样例属于开发验证集；自己导入的新图要看每次结果.json中的input.split，不能一概叫测试集。',
'- 15冻结版本独立测试：本次70张测试图片和对应预测，图上标记HELD-OUT TEST，报告明确标记“独立测试集”。',
'- 03早期运行/原理核验可能包含建库样例和旧开发预测；04早期可视化按下方逐文件清单核对，不能仅凭目录名推断。',
'- 热力图、预测掩码、叠加图和GT是同一张样本的不同表示，沿用该样本的集合身份，不算多张独立样本。','',
'## 如何核对任意图片','',
'打开同目录“图片属于哪个数据集.html”，输入完整相对路径或编号查找。不要只看005.png，因为good/005与scratch/005可能是不同集合。固定成员清单是02代码与配置/configs/splits/metal_nut-v1.json。','',
'旧实验文件保留原样，不把历史结果中的“开发”标签改成“测试”。新的说明页为历史图片提供来源索引。无法由路径或元数据确认的图明确列为待核对，不猜测。']
text='\n'.join(lines)+'\n'
(out/'图片集合说明.md').write_text(text,encoding='utf-8')
figures=[]
for f in sorted((ROOT/'04实验结果').rglob('*.png')):
    rel=f.relative_to(ROOT).as_posix(); sample=None;split=None;note=''
    m=re.search(r'metal_nut/(?:test|train)/[^/]+/\d+\.png',rel)
    if m and m.group() in membership:sample=m.group()
    if not sample:
        m=re.search(r'metal_nut_(test|train)_([^_/]+)_(\d+)\.png',f.name)
        if m:
            candidate=f'metal_nut/{m[1]}/{m[2]}/{m[3]}.png'
            if candidate in membership:sample=candidate
    if not sample:
        m=re.search(r'(?:metal_nut/|images/)((?:test|train)/[^/]+/\d+)(?:_zoom)?\.png',rel)
        if m:
            candidate='metal_nut/'+m[1]+'.png'
            if candidate in membership:sample=candidate
    if not sample:
        m=re.fullmatch(r'case_\d+_(bent|color|flip|scratch|good)_(\d+)_(full|zoom)\.png',f.name)
        if m:
            candidate=f'metal_nut/test/{m[1]}/{m[2]}.png'
            if candidate in membership:sample=candidate
    if sample:split=membership[sample];note='由样本路径或全图/放大图文件命名匹配固定清单'
    if not sample and (f.parent/'结果.json').exists():
        result=json.loads((f.parent/'结果.json').read_text(encoding='utf-8'))
        sample=result.get('input',{}).get('manifest_image')
        split=result.get('input',{}).get('split')
        note='由单图检测结果元数据确认'
    if not split and f.name=='score_distribution.png':
        split='mixed';note='44正常校准＋45开发验证的分数分布'
    if not split and f.name in ['IoU-Recall.png','FP-Recall.png','Precision与误圈占比.png','召回与误圈权衡.png']:
        split='development';note='开发验证集汇总统计图，非单张样本'
    if not split:split='unresolved';note='路径不足以确认，需查看原报告或输入清单'
    figures.append({'figure_path':rel,'image_path':sample,'split':split,'split_label':LABEL.get(split,{'mixed':'混合统计图','user_image':'用户新图（未划入固定集合）','unresolved':'待核对'}.get(split,split)),'evidence':note})
(out/'文档图片来源索引.json').write_text(json.dumps(figures,ensure_ascii=False,indent=2),encoding='utf-8')
from analyze_matched_recall import html_summary
def tr(row,kind):
    path=row['image_path'] if kind=='sample' else row['figure_path']
    url='../03原始数据/解压数据/MVTecAD/'+path if kind=='sample' else '../'+path
    note=row['defect_type'] if kind=='sample' else row['evidence']
    return f'<tr data-split="{row["split"]}"><td><a href="{html.escape(url)}">{html.escape(path)}</a></td><td>{html.escape(row["split_label"])}</td><td>{html.escape(note)}</td></tr>'
css='<style>body{max-width:1350px;margin:30px auto;padding:0 20px;font:16px/1.7 system-ui;background:#f5f8fb;color:#20384d}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{border:1px solid #d8e1e9;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#e2edf5}input,select{font:inherit;padding:10px;margin:10px 8px 16px 0}input{width:55%}a{color:#146aaa}h2{margin-top:30px}details{margin:25px 0}</style>'
filters='<input id="search" placeholder="输入color/001、good/005、报告目录名等"><select id="split"><option value="">全部集合</option>'+''.join(f'<option value="{s}">{label}</option>' for s,label in (LABEL|{'mixed':'混合统计图','user_image':'用户新图','unresolved':'待核对'}).items())+'</select>'
tables='<h2>全部335张原图的集合身份</h2><table class="filter"><tr><th>图片路径</th><th>归属</th><th>类别</th></tr>'+''.join(tr(r,'sample') for r in records)+'</table><h2>已有报告图片来源索引</h2><p>有些图是汇总统计图；待核对不等于测试集。</p><table class="filter"><tr><th>报告图片路径</th><th>归属</th><th>依据</th></tr>'+''.join(tr(r,'figure') for r in figures)+'</table>'
js='<script>function filter(){const q=document.getElementById("search").value.toLowerCase(),s=document.getElementById("split").value;document.querySelectorAll("tr[data-split]").forEach(r=>r.hidden=!(r.textContent.toLowerCase().includes(q)&&(!s||r.dataset.split===s)));}document.getElementById("search").addEventListener("input",filter);document.getElementById("split").addEventListener("change",filter);</script>'
(out/'图片属于哪个数据集.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>图片集合身份查询</title>'+css+'<h1>文档里的图片属于哪个集合</h1>'+html_summary(text)+filters+tables+js+'</html>',encoding='utf-8')
print(json.dumps({'samples':len(records),'figures':len(figures),'figure_groups':dict(Counter(r['split_label'] for r in figures))},ensure_ascii=False))
