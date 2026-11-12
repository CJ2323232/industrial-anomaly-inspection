import unittest
import numpy as np
from scripts.check_threshold_stability import count_at, aggregate
from scripts.threshold_scan_metrics import pixel_metrics


class ThresholdStabilityTests(unittest.TestCase):
    def test_strict_threshold_ties(self):
        pos=np.array([1,2,2,4],dtype=np.float32)
        neg=np.array([0,2,3],dtype=np.float32)
        self.assertEqual(count_at(pos,neg,2),{'TP':1,'FP':1,'FN':3,'TN':2})

    def test_pooled_metrics_and_normal_fp(self):
        a=pixel_metrics({'TP':9,'FP':1,'FN':1,'TN':89})|{'area_ratio':.1}
        b=pixel_metrics({'TP':0,'FP':2,'FN':0,'TN':98})|{'area_ratio':0}
        c=pixel_metrics({'TP':1,'FP':0,'FN':0,'TN':999})|{'area_ratio':.001}
        result=aggregate([a,b,c])
        self.assertAlmostEqual(result['Pixel_IoU'],10/14)
        self.assertAlmostEqual(result['macro_IoU'],(9/11+1)/2)
        self.assertEqual(result['normal_FP'],2)
        self.assertEqual(result['small_Recall'],1)


if __name__=='__main__': unittest.main()
