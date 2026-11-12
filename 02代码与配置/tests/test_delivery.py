import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from delivery_runtime import DeliveryInspector,PROFILES,ROOT,read_json

class DeliveryTests(unittest.TestCase):
    def test_default_references_frozen_thresholds_without_real_checkpoint(self):
        p=read_json(PROFILES)['profiles']['r18_reference']
        ref=read_json(ROOT/'02代码与配置/configs/decisions/metal_nut-development-reference-v1.json')
        self.assertEqual(p['image_threshold'],ref['image_threshold'])
        self.assertEqual(p['provisional_pixel_threshold'],ref['pixel_threshold'])
        config={'backbone':p['backbone'],'image_size':p['input_size']}
        # Real resolver hashing/import is covered by byte-fixture artifact tests.
        with patch('delivery_runtime.resolve_artifact',return_value=(ROOT/'synthetic-model.pt',config,'fixture')) as resolve:
            engine=DeliveryInspector()
        self.assertIsNone(engine.model);resolve.assert_called_once()
    def test_output_escape_rejected_before_artifact_access(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('delivery_runtime.resolve_artifact',side_effect=AssertionError('must not touch artifacts')):
                with self.assertRaisesRegex(ValueError,'输出目录'):
                    DeliveryInspector(output_dir=Path(d))
    def test_unknown_profile_is_error(self):
        with self.assertRaisesRegex(ValueError,'未知版本'):DeliveryInspector('not-a-profile')

if __name__=='__main__':unittest.main()

