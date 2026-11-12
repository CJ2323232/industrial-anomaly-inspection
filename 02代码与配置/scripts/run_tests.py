"""Test entry points: basic=stdlib only, unit=CPU numerical suite, integration=opt-in artifacts."""
from pathlib import Path
import argparse,json,os,sys,tempfile,unittest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",choices=["basic","unit","integration"],default="basic")
    parser.add_argument("--artifact-root",type=Path)
    parser.add_argument("--json",type=Path,help="Optional explicit local audit output")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];code=root/"02代码与配置"
    # All test scratch files stay inside this checkout.
    scratch=root/".cache/test-tmp";scratch.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(scratch)
    os.environ["TMP"]=str(scratch);os.environ["TEMP"]=str(scratch)
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    os.environ["IAI_RUN_MODEL_INTEGRATION"]="1" if args.suite=="integration" else "0"
    if args.artifact_root:os.environ["IAI_ARTIFACT_ROOT"]=str(args.artifact_root.resolve())
    sys.path.insert(0,str(code))
    loader=unittest.TestLoader()
    if args.suite=="basic":
        suite=unittest.TestSuite(loader.discover(str(code/"tests"),pattern=name)
                for name in ["test_model_artifacts.py","test_saved_score_example.py","test_model_release.py"])
    elif args.suite=="integration":
        suite=loader.discover(str(code/"tests"),pattern="test_delivery_integration.py")
    else:
        suite=loader.discover(str(code/"tests"),pattern="test_*.py")
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(suite=args.suite,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                skipped=[dict(test=str(t),reason=reason) for t,reason in result.skipped],passed=result.wasSuccessful(),
                gpu_enabled=False,real_model_tests_enabled=args.suite=="integration")
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if result.wasSuccessful() else 1

if __name__=="__main__":raise SystemExit(main())

