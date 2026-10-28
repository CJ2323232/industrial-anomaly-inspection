import unittest
from scripts.run_saved_score_example import metrics,validate_rows
from scripts.run_public_score_summary import analyze_summary

class SavedScoreTests(unittest.TestCase):
    def test_perfect_order_and_strict_threshold(self):
        result=metrics([0,0,1,1],[0,1,2,3],2)
        self.assertEqual((result["TP"],result["FN"],result["FP"]),(1,1,0))
        self.assertEqual(result["AUROC"],1);self.assertEqual(result["AP"],1)
        self.assertAlmostEqual(result["F1"],2/3)
    def test_ties_grouped_and_permutation_invariant(self):
        a=metrics([1,0,1,0],[2,2,2,2],2)
        b=metrics([0,1,0,1],[2,2,2,2],2)
        self.assertEqual(a,b);self.assertEqual(a["AUROC"],.5);self.assertEqual(a["AP"],.5)
        self.assertIsNone(a["Precision"]);self.assertEqual(a["Recall"],0)
    def test_inverted_order_matches_hand_calculation(self):
        result=metrics([1,1,0,0],[1,0,3,2],1)
        self.assertEqual(result["AUROC"],0)
        self.assertAlmostEqual(result["AP"],(1/3+2/4)/2)
    def test_invalid_alignment_and_values_rejected(self):
        row=dict(image_path="x",split="development",label=0,scores=dict(A=1,B=1,C=1))
        with self.assertRaisesRegex(ValueError,"重复"):validate_rows([row,row])
        with self.assertRaisesRegex(ValueError,"有限"):metrics([0,1],[1,float("nan")],0)
        with self.assertRaisesRegex(ValueError,"校准"):validate_rows([dict(row,split="normal_calibration",label=1)])
    def test_public_fixture_recomputes_frozen_results(self):
        summary=analyze_summary()
        self.assertTrue(all(c["match"] for c in summary["checks"]))
        self.assertEqual(len(summary['checks']),30)
        self.assertEqual([summary["groups"][g]["FP"] for g in "ABC"],[1,1,0])
        self.assertEqual(summary["image_reads"],0)

if __name__=="__main__":unittest.main()

