"""Recompute aggregate metrics from unpaired score-frequency summaries, stdlib only."""
from pathlib import Path
import argparse,csv,json,math,hashlib
from datetime import datetime
try:
    from .run_saved_score_example import metrics,read_json,ROOT
except ImportError:
    from run_saved_score_example import metrics,read_json,ROOT
SOURCE=ROOT/"05项目文档/最终交付/公开复核摘要/score_frequencies.json"

def analyze_summary(path=SOURCE):
    data=read_json(path);results={};checks=[]
    if set(data["groups"])!=set("ABC"):raise ValueError("需要A/B/C三组")
    for group,part in data["groups"].items():
        labels=[];scores=[];cal=[];seen=set()
        for row in part["distribution"]:
            if set(row)!={"split","label","score","count"}:raise ValueError("摘要不允许图像标识或额外字段")
            split,y,score,n=(row[k] for k in ("split","label","score","count"))
            if split not in ("development","normal_calibration") or type(y)is not int or y not in (0,1) or type(n)is not int or not 1<=n<=89:
                raise ValueError("非法计数或标签")
            if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score):raise ValueError("非法分数")
            key=(split,y,score)
            if key in seen:raise ValueError("重复频次桶")
            seen.add(key)
            if split=="normal_calibration":
                if y!=0:raise ValueError("校准必须正常")
                cal.extend([score]*n)
            else:labels.extend([y]*n);scores.extend([score]*n)
        if (len(labels),sum(labels),len(cal))!=(45,37,44):raise ValueError("冻结摘要样本量不符")
        threshold=part["threshold"];m=metrics(labels,scores,threshold)
        m["calibration_exceedances"]=sum(v>threshold for v in cal)
        for key,value in part["expected"].items():
            if not math.isclose(m[key],value,abs_tol=1e-12,rel_tol=1e-12):raise ValueError("指标不一致："+group+"/"+key)
            checks.append(dict(group=group,metric=key,match=True))
        m["threshold"]=threshold;results[group]=m
    return dict(groups=results,checks=checks,image_reads=0,model_forwards=0,threshold_selection=False,
        new_independent_evidence=False,source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        scope="45开发图指标与44正常校准图分数摘要；没有逐图配对或错误定位信息")

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path);args=parser.parse_args()
    try:
        result=analyze_summary()
        out=args.output or ROOT/"04实验结果/29公开分数复核"/datetime.now().strftime("summary_%Y%m%d_%H%M%S_%f")
        out.mkdir(parents=True,exist_ok=False)
        (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        lines=["# 公开频次摘要复核","","只复核已有开发数据，不读取图片、不推理、不选择阈值。","","|组|FP/8|FN/37|Precision|Recall|F1|AUROC|AP|","|---|---:|---:|---:|---:|---:|---:|---:|"]
        for g,m in result["groups"].items():
            lines.append(f"|{g}|{m['FP']}|{m['FN']}|{m['Precision']:.6f}|{m['Recall']:.6f}|{m['F1']:.6f}|{m['AUROC']:.6f}|{m['AP']:.6f}|")
        lines+=["","30项既有指标核对通过。只输出汇总，不公开图片编号、逐图判断、配对关系或错误案例。","原始精度分数按组/划分/标签聚合计数；个别桶可只有一张图，数据不应被描述为隐私匿名化保证。","C仍是开发集候选；本输出不是新增独立测试。"]
        (out/"报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        print(json.dumps(dict(output=str(out.resolve()),checks=len(result["checks"]),new_independent_evidence=False),ensure_ascii=False))
    except (OSError,ValueError,KeyError,TypeError) as exc:parser.exit(2,str(exc)+"\n")
if __name__=="__main__":main()

