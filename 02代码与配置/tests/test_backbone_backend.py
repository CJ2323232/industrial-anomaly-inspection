import unittest
import torch
from scripts.backbone_backend import select_feature_weights


class PretrainedMappingTests(unittest.TestCase):
    def test_strict_feature_mapping_excludes_only_unused_tail(self):
        target = {'conv1.weight': torch.zeros(3, 3), 'layer3.weight': torch.zeros(4, 3)}
        source = {**target, 'layer4.weight': torch.zeros(1), 'fc.weight': torch.zeros(2)}
        selected, unused = select_feature_weights(target, source)
        self.assertEqual(set(selected), set(target))
        self.assertEqual(unused, ['fc.weight', 'layer4.weight'])

    def test_missing_feature_weight_rejected(self):
        with self.assertRaises(ValueError):
            select_feature_weights({'layer3.weight': torch.zeros(4)}, {})

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            select_feature_weights({'layer3.weight': torch.zeros(4)}, {'layer3.weight': torch.zeros(3)})

    def test_unexpected_ignored_feature_rejected(self):
        with self.assertRaises(ValueError):
            select_feature_weights({'conv1.weight': torch.zeros(4)},
                                   {'conv1.weight': torch.zeros(4), 'layer2.wrong_name': torch.zeros(1)})


if __name__ == '__main__':
    unittest.main()
