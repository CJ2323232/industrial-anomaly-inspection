import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from delivery_topk import topk_image_score
class DeliveryTopKTests(unittest.TestCase):
    def test_averages_unsmoothed_highest_eleven(self):
        values=np.arange(1024,dtype=np.float32)[::-1].reshape(32,32)
        score,detail=topk_image_score(values)
        self.assertEqual(score,1018.)
        self.assertEqual(detail['top_indices'],list(range(11)))
    def test_ties_are_stable_and_float64_accumulation(self):
        values=np.zeros(1024,dtype=np.float32);values[:15]=np.float32(.1)
        score,detail=topk_image_score(values)
        self.assertEqual(detail['top_indices'],list(range(11)))
        self.assertEqual(score,float(np.float64(np.float32(.1))))
    def test_rejects_changed_grid_k_and_nonfinite(self):
        for values,k in [(np.zeros(100,dtype=np.float32),11),(np.zeros(1024,dtype=np.float32),10),(np.full(1024,np.nan,dtype=np.float32),11)]:
            with self.assertRaises(ValueError):topk_image_score(values,k)
if __name__=='__main__':unittest.main()
