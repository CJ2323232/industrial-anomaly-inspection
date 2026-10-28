"""Recompute published development scores with the Python standard library only.

No image reads, model imports, network, fitting or threshold selection.
"""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import hashlib
import html
import json
import math

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "05项目文档/最终交付/评分实验只读证据"
GROUPS = ("A", "B", "C")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("非有限JSON数值："+x)))


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(labels, scores, threshold):
    if not labels or len(labels) != len(scores):
        raise ValueError("标签与分数必须非空且一一对应")
    if any(type(x) is not int or x not in (0,1) for x in labels):
        raise ValueError("标签必须是整数0/1")
    if not math.isfinite(threshold) or any(isinstance(x,bool) or not isinstance(x,(float,int))
                                         or not math.isfinite(x) for x in scores):
        raise ValueError("分数和阈值必须有限")
    predictions = [int(s > threshold) for s in scores]
    tp=sum(y==1 and p==1 for y,p in zip(labels,predictions))
    fp=sum(y==0 and p==1 for y,p in zip(labels,predictions))
    fn=sum(y==1 and p==0 for y,p in zip(labels,predictions))
    tn=sum(y==0 and p==0 for y,p in zip(labels,predictions))
    positives=[s for y,s in zip(labels,scores) if y==1]
    negatives=[s for y,s in zip(labels,scores) if y==0]
    auc = (sum((p>n)+0.5*(p==n) for p in positives for n in negatives)
           / (len(positives)*len(negatives))) if positives and negatives else None
    # Non-interpolated average precision; equal scores enter as a single block.
    ap=None
    if positives:
        order=sorted(zip(scores,labels),reverse=True)
        i=0;seen=0;true=0;ap=0.0
        while i<len(order):
            j=i
            while j<len(order) and order[j][0]==order[i][0]:j+=1
            gained=sum(y for _,y in order[i:j]);true+=gained;seen+=j-i
            ap += gained/len(positives)*true/seen
            i=j
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,Recall=ratio(tp,tp+fn),FPR=ratio(fp,fp+tn),
                Precision=ratio(tp,tp+fp),F1=ratio(2*tp,2*tp+fp+fn),AUROC=auc,AP=ap)


def validate_rows(rows):
    if not isinstance(rows,list) or not rows:raise ValueError("逐图记录必须为非空列表")
    seen=set()
    for row in rows:
        name=row.get("image_path")
        if not isinstance(name,str) or not name or name in seen:
            raise ValueError("图片标识缺失或重复")
        seen.add(name)
        if row.get("split") not in ("normal_calibration","development"):
            raise ValueError("本示例只接受已公开的正常校准/开发记录")
        label=row.get("label")
        if type(label) is not int or label not in (0,1):raise ValueError("标签必须为整数0/1")
        if row["split"]=="normal_calibration" and label!=0:
            raise ValueError("正常校准集不允许异常标签")
        if set(row.get("scores",{}))!=set(GROUPS):raise ValueError("A/B/C分数未对齐")
        for score in row["scores"].values():
            if isinstance(score,bool) or not isinstance(score,(float,int)) or not math.isfinite(score):
                raise ValueError("分数必须有限")


def analyze(evidence=EVIDENCE):
    evidence=Path(evidence)
    rows=read_json(evidence/"paired_scores.json");validate_rows(rows)
    development=[r for r in rows if r["split"]=="development"]
    calibration=[r for r in rows if r["split"]=="normal_calibration"]
    if (len(development),sum(r["label"] for r in development),len(calibration))!=(45,37,44):
        raise ValueError("此冻结示例要求45开发图（37异常、8正常）及44正常校准图")
    results={};per_image=[];checks=[]
    sources=["paired_scores.json"]
    for group in GROUPS:
        config=read_json(evidence/group/"config.json")
        if config["group"]!=group:raise ValueError("评分配置组名不匹配")
        threshold=config["analysis_image_threshold"]
        result=metrics([r["label"] for r in development],[r["scores"][group] for r in development],threshold)
        result["image_threshold"]=threshold
        result["calibration_exceedances"]=sum(r["scores"][group]>threshold for r in calibration)
        result["calibration_range"]=[min(r["scores"][group] for r in calibration),max(r["scores"][group] for r in calibration)]
        result["FP_images"]=[r["image_path"] for r in development if r["label"]==0 and r["scores"][group]>threshold]
        result["FN_images"]=[r["image_path"] for r in development if r["label"]==1 and r["scores"][group]<=threshold]
        expected=read_json(evidence/group/"metrics.json")
        for key in ("TP","FP","FN","TN","Recall","FPR","Precision","AUROC","AP","calibration_exceedances"):
            ok=math.isclose(result[key],expected[key],rel_tol=1e-12,abs_tol=1e-12)
            checks.append(dict(group=group,metric=key,match=ok))
            if not ok:raise ValueError(f"{group}/{key}与已公开结果不一致")
        for key in ("FP_images","FN_images"):
            if sorted(result[key])!=sorted(expected[key]):raise ValueError(f"{group}/{key}不一致")
        for row in rows:
            predicted=int(row["scores"][group]>threshold)
            if row.get("predictions",{}).get(group)!=predicted:
                raise ValueError("已有逐图判断与冻结阈值不一致："+row["image_path"])
            per_image.append(dict(image_path=row["image_path"],split=row["split"],label=row["label"],
                group=group,score=row["scores"][group],threshold=threshold,prediction=predicted,
                correct=predicted==row["label"]))
        sources.extend([group+"/config.json",group+"/metrics.json"])
        results[group]=result
    fingerprints={name:hashlib.sha256((evidence/name).read_bytes()).hexdigest() for name in sources}
    return dict(scope="Recomputed saved development results; NOT new independent evidence",
                development_count=45,calibration_count=44,groups=results,checks=checks,
                image_reads=0,model_forwards=0,training_runs=0,threshold_selection=False,
                sources_sha256=fingerprints),per_image


def report_markdown(summary):
    lines=["# 无模型 A/B/C 公开分数复核",
           "","> 本报告仅重算已公开的45张开发图及44张正常校准图分数。没有新图片、模型推理或阈值选择；不能作为C的新独立可靠性证据。",
           "","|组|固定图像阈值|误报/8|漏检/37|Precision|Recall|F1|AUROC|AP|",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group,m in summary["groups"].items():
        lines.append(f"|{group}|{m['image_threshold']:.10f}|{m['FP']}|{m['FN']}|{m['Precision']:.6f}|{m['Recall']:.6f}|{m['F1']:.6f}|{m['AUROC']:.6f}|{m['AP']:.6f}|")
    lines += ["","A：原始邻居加权；B：最大Patch距离；C：已保存的Top-11平均距离。",
              "阈值直接读取原config，不从本次数据重新挑选。44张校准图仅用于报告其已存分数范围和超过冻结阈值数量。",
              "","## 指标口径",
              "- Recall=TP/(TP+FN)：异常图片检出比例。",
              "- Precision=TP/(TP+FP)：被判异常的图片中真实异常比例。",
              "- FPR=FP/(FP+TN)：正常图片误报比例。",
              "- F1=2TP/(2TP+FP+FN)：Precision与Recall的调和平均。",
              "- AUROC：异常分数高于正常分数的配对比例，同分计0.5。",
              "- AP：按分数降序、同分成组计算的非插值平均精确率。",
              "- 这里只处理图像级分数；没有GT Mask，不计算Pixel IoU，也不重算Top-K。",
              "","## 结果边界",
              "A/B为1张误报、0漏检；C为0误报、0漏检，三组AUROC/AP均约1。这是既有开发结果，不是新增测试成绩。",
              "C相对A/B只在该开发集少1张误报，不能据此承诺新弱/小缺陷不会漏检。",
              "","逐图原始分数、固定阈值和判断见per_image.csv；完整指标及来源SHA见summary.json。",
              "复核通过项："+str(len(summary["checks"]))+"；没有读取路径指向的图片或原始Patch数组。"]
    return "\n".join(lines)+"\n"


def write_outputs(output, summary, rows):
    output=Path(output)
    if output.exists():raise ValueError("输出目录已存在；请选择新目录，避免覆盖既有结果")
    output.mkdir(parents=True)
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (output/"per_image.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    markdown=report_markdown(summary)
    (output/"报告.md").write_text(markdown,encoding="utf-8")
    # A preformatted, escaped view avoids third-party rendering dependencies.
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>公开分数复核</title><style>body{max-width:1150px;margin:32px auto;padding:20px;background:#f8fafc;color:#123}pre{white-space:pre-wrap;font:16px/1.8 system-ui}</style><pre>'+html.escape(markdown)+'</pre></html>'
    (output/"报告.html").write_text(page,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir",type=Path,default=EVIDENCE)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    try:
        summary,rows=analyze(args.evidence_dir)
        output=args.output or ROOT/"04实验结果/29公开分数复核"/datetime.now().strftime("saved_scores_%Y%m%d_%H%M%S_%f")
        write_outputs(output,summary,rows)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        parser.exit(2,str(exc)+"\n")
    print(json.dumps(dict(output=str(output.resolve()),verified_checks=len(summary["checks"]),
        groups={g:{k:m[k] for k in ("FP","FN","AUROC","AP")} for g,m in summary["groups"].items()},
        new_independent_validation=False),ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

