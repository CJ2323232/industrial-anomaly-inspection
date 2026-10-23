"""Synthetic smoothing checks, including the real reference generator/model."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from scripts import compare_smoothing as runner
from scripts.baseline_utils import file_sha256
from scripts.run_trusted_baseline import make_model, predict, state_sha256
from scripts.smoothing_backend import use_smoothing


class SmoothingTests(unittest.TestCase):
    def test_reference_gaussian_sigma2_is_narrower_than_sigma4(self):
        from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator
        impulse=torch.zeros(1,1,64,64)
        impulse[0,0,32,32]=1
        maps={s:AnomalyMapGenerator(sigma=s)(impulse) for s in (4,2)}
        self.assertGreater(maps[2][0,0,32,32],maps[4][0,0,32,32])
        x=torch.arange(64)-32
        radius=x[:,None].square()+x[None,:].square()
        moments={s:float((a[0,0]*radius).sum()/a.sum()) for s,a in maps.items()}
        self.assertGreater(moments[4],moments[2])
        for value in maps.values():
            self.assertAlmostEqual(float(value.sum()),1.0,places=5)

    def test_iou_gain_does_not_pass_when_pixel_recall_falls(self):
        def quality(iou,recall):
            return {'pixel_metrics':{'recall':recall},'normal_false_positive_pixels':{'rate':.005},
                    'anomaly_macro':{'pixel_IoU':iou},'area_groups':{'0.1_to_lt_1_percent':
                    {'pixel_metrics':{'recall':recall},'macro':{'pixel_IoU':iou}}}}
        result=runner.decision_checks({4:quality(.3,.96),2:quality(.4,.90)}, {'image_prediction_changes':0})
        self.assertTrue(result['checks']['anomaly_macro_IoU_improved'])
        self.assertFalse(result['checks']['pooled_defect_pixel_recall_not_lower'])
        self.assertFalse(result['all_quality_conditions_met'])
        self.assertFalse(result['new_sigma_automatically_adopted'])

    def test_pair_labels_express_sigma_instead_of_resolution(self):
        row={'image_path':'synthetic','label':1,'pred_label':1,'defect_type':'scratch',
             'area_ratio':.25,'raw_score':2.0,'pixel_counts':{'TP':1,'FP':1,'FN':0,'TN':2},
             'pixel_AP':.8,'pixel_AUROC':.9}
        result=runner.relabel_pairs([row],[copy.deepcopy(row)])
        paired=result['per_image'][0]
        self.assertIn('pixel_IoU_sigma4',paired)
        self.assertIn('pixel_IoU_sigma2',paired)
        self.assertFalse(any(k.endswith(('_256','_512')) for k in paired))
        self.assertEqual(result['image_prediction_changes'],0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_real_model_changes_map_only_and_restores_after_error(self):
        torch.manual_seed(22)
        model=make_model({'backbone':'resnet18','layers':['layer2','layer3'],'num_neighbors':9},pretrained=False).cuda().eval()
        model.requires_grad_(False)
        model.memory_bank=torch.randn(13,384,device='cuda')
        image=torch.randn(1,3,256,256,device='cuda')
        original=model.anomaly_map_generator
        frozen=state_sha256(model.state_dict())
        with torch.no_grad():reference=model(image)
        with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
            with use_smoothing(model,2):
                with torch.no_grad():candidate=model(image)
                torch.testing.assert_close(reference.pred_score,candidate.pred_score,rtol=0,atol=0)
                self.assertFalse(torch.allclose(reference.anomaly_map,candidate.anomaly_map))
                raise RuntimeError('synthetic failure')
        self.assertIs(model.anomaly_map_generator,original)
        self.assertEqual(state_sha256(model.state_dict()),frozen)
        with torch.no_grad():restored=model(image)
        torch.testing.assert_close(reference.anomaly_map,restored.anomaly_map,rtol=0,atol=0)
        for value in (0,-1,2.5,True):
            with self.assertRaises(ValueError):
                with use_smoothing(model,value):pass
        model.cpu()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_synthetic_full_runner_recalibrates_and_generates_reports(self):
        from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator
        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor=torch.nn.Identity()
                self.anomaly_map_generator=AnomalyMapGenerator(sigma=4)
                self.register_buffer('memory_bank',torch.arange(26,dtype=torch.float32).reshape(13,2))
            def forward(self,image):
                value=image.mean(dim=(1,2,3))+self.memory_bank[0,0]*0
                raw=value[:,None,None,None].expand(-1,1,64,64)
                return SimpleNamespace(pred_score=value,anomaly_map=self.anomaly_map_generator(raw))
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root/'.cache/tmp',prefix='smoothing_test_') as temp:
            temp=Path(temp)
            baseline=temp/'baseline';baseline.mkdir()
            (baseline/'model.pt').write_text('synthetic placeholder',encoding='utf-8')
            policy=temp/'policy.json';policy.write_text('{}',encoding='utf-8')
            data=temp/'data'
            normal='metal_nut/test/good/000.png'
            defect='metal_nut/test/scratch/000.png'
            calibration='metal_nut/train/good/cal.png'
            values={normal:.5,defect:2.,calibration:1.}
            records={p:{'label':int(p==defect),'defect_type':'scratch' if p==defect else 'good',
                        'width':4,'height':4,'image_sha256':'synthetic','mask_sha256':'synthetic' if p==defect else None} for p in values}
            for p in (normal,defect):
                path=data/p;path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.zeros((4,4,3),dtype=np.uint8)).save(path)
            manifest={'manifest_sha256':'synthetic','splits':{'fit':['NOT_READ'], 'normal_calibration':[calibration],
                      'development':[normal,defect],'held_out':['NEVER_READ']},'files':records}
            config={'image_size':[256,256],'image_quantile':.95,'pixel_quantile':.995}
            original_model=Detector().eval()
            original_rows=[]
            original_maps={}
            for p,value in values.items():
                score,a=predict(original_model,torch.full((3,64,64),value),(4,4))
                relative=Path('maps')/Path(p).with_suffix('.npz')
                destination=baseline/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(destination,anomaly_map=a)
                original_rows.append({'image_path':p,'raw_score':score,'map_path':relative.as_posix()})
                original_maps[p]=a
            cal=[r for r in original_rows if r['image_path']==calibration]
            dev=[r for r in original_rows if r['image_path']!=calibration]
            source={'model.pt':file_sha256(baseline/'model.pt')}
            source.update({r['map_path']:file_sha256(baseline/r['map_path']) for r in original_rows})
            report={'config':config,'thresholds':{'pixel_threshold':float(np.quantile(original_maps[calibration],.995,method='linear'))},
                    'frozen_backbone_state_sha256':state_sha256(original_model.feature_extractor.state_dict())}
            def loader(path,device):
                return Detector().to(device).eval(),{'model_config':copy.deepcopy(config),'manifest_sha256':'synthetic'}
            def mask(data_root,record):
                a=np.zeros((4,4),dtype=np.uint8)
                if record['label']:a[1:3,1:3]=1
                return a
            args=SimpleNamespace(baseline_dir=baseline,data_root=data,policy=policy,warmup=1,repeats=1)
            with patch.object(runner,'load_checkpoint',side_effect=loader), \
                 patch('scripts.compare_resolution.make_transform',return_value=None), \
                 patch('scripts.compare_resolution.load_tensor',side_effect=lambda d,p,t:torch.full((3,64,64),values[p])), \
                 patch('scripts.compare_resolution.load_mask',side_effect=mask), \
                 patch.object(runner,'make_transform',return_value=None), \
                 patch.object(runner,'load_tensor',side_effect=lambda d,p,t:torch.full((3,64,64),values[p])), \
                 patch.object(runner,'load_mask',side_effect=mask):
                result=runner.run_experiment(args,(source,report,manifest,cal,dev,1.,{}),temp/'output')
            self.assertEqual(result['development_images'],2)
            self.assertTrue(result['image_scores_and_decisions_unchanged'])
            self.assertTrue(result['model_state_unchanged'])
            self.assertEqual(result['held_out_images_opened'],0)
            self.assertFalse(result['new_model_or_bank_fitted'])
            self.assertFalse(result['decision']['new_sigma_automatically_adopted'])
            for sigma in ('sigma4','sigma2'):
                self.assertEqual(result['thresholds'][sigma]['image_threshold'],1.)
                self.assertEqual(result['performance'][sigma]['samples'],2)
                self.assertGreater(result['performance'][sigma]['mean_ms'],0)
                self.assertTrue((temp/f'output/{sigma}/calibration.json').exists())
            self.assertEqual(len(list((temp/'output/可视化').rglob('*.png'))),1)
            self.assertTrue((temp/'output/结果摘要.txt').exists())
            self.assertEqual(json.loads((temp/'output/status.json').read_text())['status'],'completed')
            self.assertFalse(list((temp/'output').rglob('*.pt')))
            self.assertEqual(file_sha256(baseline/'model.pt'),source['model.pt'])


if __name__=='__main__':unittest.main()
