import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.inspection_runtime import InputGuard, threshold_result, visualizations

class InspectionTests(unittest.TestCase):
    def test_strict_threshold_and_independent_image_result(self):
        scores=np.array([[1.,2.],[3.,4.]],dtype=np.float32)
        anomalous,mask=threshold_result(1.,scores,1.,2.)
        self.assertFalse(anomalous)
        np.testing.assert_array_equal(mask,[[False,False],[True,True]])

    def test_nan_prediction_rejected(self):
        with self.assertRaises(ValueError):
            threshold_result(float('nan'),np.zeros((2,2)),1,1)

    def test_heatmap_does_not_change_scores(self):
        scores=np.array([[0.,1.],[2.,3.]],dtype=np.float32);before=scores.copy()
        image=Image.new('RGB',(2,2),'gray')
        _,overlay,mask=visualizations(image,scores,2.)
        np.testing.assert_array_equal(scores,before)
        self.assertEqual(np.asarray(mask).tolist(),[[0,0],[0,255]])
        np.testing.assert_array_equal(np.asarray(overlay)[0],np.asarray(image)[0])

    def test_sealed_path_rejected_before_content_read(self):
        root=Path.cwd()/'synthetic-not-created'
        manifest={'splits':{'held_out':['sealed.png'],'development':['dev.png']},
                  'files':{'sealed.png':{'image_sha256':'sealed'},'dev.png':{'image_sha256':'dev'}}}
        guard=InputGuard(manifest,root)
        with patch.object(Path,'read_bytes',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'保留测试'):
                guard.load(root/'sealed.png')

if __name__=='__main__':unittest.main()
