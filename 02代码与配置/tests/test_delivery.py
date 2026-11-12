import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from delivery_runtime import DeliveryInspector,PROFILES,ROOT,read_json
class DeliveryTests(unittest.TestCase):
    def test_default_references_existing_frozen_thresholds(self):
        p=read_json(PROFILES)['profiles']['r18_reference']
        ref=read_json(ROOT/'02代码与配置/configs/decisions/metal_nut-development-reference-v1.json')
        self.assertEqual(p['image_threshold'],ref['image_threshold'])
        self.assertEqual(p['provisional_pixel_threshold'],ref['pixel_threshold'])
        engine=DeliveryInspector();self.assertIsNone(engine.model)
    def test_output_escape_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'输出目录'):
                DeliveryInspector(output_dir=Path(d))
    def test_unknown_profile_is_error(self):
        with self.assertRaisesRegex(ValueError,'未知版本'):DeliveryInspector('not-a-profile')
if __name__=='__main__':unittest.main()
