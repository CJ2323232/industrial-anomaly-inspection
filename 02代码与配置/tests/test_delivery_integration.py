"""Opt-in real artifact loading, no dataset inference."""
import os,sys,unittest
from pathlib import Path

@unittest.skipUnless(os.environ.get("IAI_RUN_MODEL_INTEGRATION")=="1",
                    "真实模型集成测试未启用；基础测试不需要权重。用 run_tests.py --suite integration 显式启用")
class ModelIntegrationTests(unittest.TestCase):
    def test_registered_models_load_and_validate_memory_bank(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
        from delivery_runtime import DeliveryInspector
        for profile in ("r18_reference","wrn50_a","wrn50_c_experimental"):
            with self.subTest(profile=profile):
                inspector=DeliveryInspector(profile,device="cpu",artifact_root=os.environ.get("IAI_ARTIFACT_ROOT"))
                inspector.load_model()
                self.assertEqual(list(inspector.model.memory_bank.shape),inspector.policy["memory_bank_shape"])
                del inspector

if __name__=="__main__":unittest.main()

