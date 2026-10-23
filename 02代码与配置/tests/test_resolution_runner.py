"""Synthetic integration checks; never opens the actual MVTec dataset."""
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

from scripts import compare_resolution as runner
from scripts.baseline_utils import file_sha256
from scripts.run_trusted_baseline import state_sha256


class SyntheticDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Identity()
        self.register_buffer('memory_bank', torch.arange(26,dtype=torch.float32).reshape(13,2))
        self.embedding_store = []

    def forward(self, image):
        value = image.mean(dim=(1,2,3)) + self.memory_bank[0,0] * 0
        return SimpleNamespace(pred_score=value, anomaly_map=value[:,None,None,None].expand(-1,1,4,4))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for synthetic resource checks')
class ResolutionRunnerTests(unittest.TestCase):
    def test_real_reference_model_supports_both_synthetic_resolutions(self):
        from scripts.run_trusted_baseline import make_model, predict
        torch.manual_seed(31)
        model=make_model({'backbone':'resnet18','layers':['layer2','layer3'],'num_neighbors':9},pretrained=False).cuda()
        model.requires_grad_(False)
        original=state_sha256(model.feature_extractor.state_dict())
        for size,expected in ((256,1024),(512,4096)):
            image=torch.randn(3,size,size,device='cuda')
            model.train()
            model.feature_extractor.eval()
            with torch.no_grad():
                embedding=model(image.unsqueeze(0))
            self.assertEqual(tuple(embedding.shape),(expected,384))
            model.memory_bank=embedding[:9].clone()
            model.embedding_store.clear()
            model.eval()
            score,score_map=predict(model,image,(40,40))
            self.assertTrue(np.isfinite(score))
            self.assertEqual(score_map.shape,(40,40))
            self.assertTrue(np.isfinite(score_map).all())
            self.assertEqual(state_sha256(model.feature_extractor.state_dict()),original)
            del image,embedding
        model.cpu()

    def test_exact_bank_cardinality_and_seed_reproducibility(self):
        generator = torch.Generator().manual_seed(812)
        data = torch.randn(17,384,generator=generator).cuda()
        original = data.clone()
        bank, indices, info = runner.select_bank(data,9,seed=0)
        bank2, indices2, info2 = runner.select_bank(data,9,seed=0)
        self.assertEqual(len(set(indices)),9)
        self.assertEqual(indices,indices2)
        torch.testing.assert_close(bank,data[indices])
        torch.testing.assert_close(data,original)
        self.assertEqual(info['bank_sha256'],info2['bank_sha256'])
        self.assertEqual(info['vectors'],9)
        with self.assertRaises(ValueError):
            runner.select_bank(data,18)

    def test_fit_feature_staging_clears_gpu_store_and_counts_candidates(self):
        class FitDetector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor=torch.nn.Identity()
                self.register_buffer('memory_bank',torch.empty(0))
                self.embedding_store=[]
            def forward(self,x):
                value=x.mean(dim=(1,2,3))
                z=torch.arange(4*384,device=x.device).float().reshape(4,384)/1000
                out=torch.cat([z + v for v in value],dim=0)
                self.embedding_store.append(out)
                return out
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root/'.cache/tmp',prefix='resolution_fit_test_') as temp:
            model=FitDetector()
            config={'image_size':[16,16],'fit_batch_size':2}
            manifest={'splits':{'fit':['a','b','c']}}
            with patch.object(runner,'make_transform',return_value=None), patch.object(runner,'load_tensor',side_effect=lambda d,p,t:torch.full((3,16,16),float('abc'.index(p)))):
                report=runner.build_512(model,config,manifest,Path(temp),Path(temp),3)
            self.assertEqual(report['candidate_feature_count'],12)
            self.assertEqual(report['bank_shape'],[3,384])
            self.assertEqual(model.embedding_store,[])
            self.assertFalse(model.training)
            self.assertGreater(report['peak_allocated_bytes'],0)
            model.cpu()

    def test_synthetic_full_runner_writes_metrics_timings_checkpoint_and_visuals(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root/'.cache/tmp',prefix='resolution_runner_test_') as temp:
            temp=Path(temp)
            baseline=temp/'baseline'
            baseline.mkdir()
            (baseline/'model.pt').write_text('synthetic placeholder',encoding='utf-8')
            data=temp/'data'
            normal='metal_nut/test/good/000.png'
            defect='metal_nut/test/scratch/000.png'
            calibration='metal_nut/train/good/cal.png'
            paths=[normal,defect]
            records={p:{'label':int(p==defect),'defect_type':'scratch' if p==defect else 'good',
                        'width':4,'height':4,'image_sha256':'synthetic','mask_sha256':'synthetic' if p==defect else None} for p in paths+[calibration]}
            for p in paths:
                image_path=data/p
                image_path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.zeros((4,4,3),dtype=np.uint8)).save(image_path)
            manifest={'manifest_sha256':'synthetic','splits':{'fit':['synthetic_fit'], 'normal_calibration':[calibration],
                        'development':paths,'held_out':['NEVER_OPEN']},'files':records}
            configs={'image_size':[256,256],'fit_batch_size':4,'image_quantile':.95,'coreset_ratio':.01}
            model=SyntheticDetector()
            original={'config':configs,'frozen_backbone_state_sha256':state_sha256(model.feature_extractor.state_dict()),
                      'thresholds':{'pixel_threshold':1.0}}
            identities={'model.pt':file_sha256(baseline/'model.pt')}
            cal=[{'image_path':calibration,'raw_score':1.0}]
            old=[{'image_path':normal,'raw_score':.5},{'image_path':defect,'raw_score':2.0}]
            info={'status':'synthetic','held_out_images_opened':0}
            values={normal:.5,defect:2.0,calibration:1.0}
            active_models=[]
            def loader(path,device):
                result=SyntheticDetector()
                if path.parent.name=='512':
                    saved=torch.load(path,map_location='cpu',weights_only=True)
                    result.load_state_dict(saved['model_state_dict'])
                else:
                    saved={'model_config':copy.deepcopy(configs),'manifest_sha256':'synthetic'}
                    active_models.append(result)
                    def check_residency(module, image):
                        if image[0].is_cuda:
                            self.assertEqual(sum(m.memory_bank.is_cuda for m in active_models),1)
                    result.register_forward_pre_hook(check_residency)
                return result.to(device).eval(),saved
            def build(model,config,manifest,data_root,output,bank_count):
                model.memory_bank=model.memory_bank + .01
                return {'effective_sampling_ratio':.25,'candidate_feature_count':52,'bank_shape':[13,2],
                        'bank_bytes':104,'peak_allocated_bytes':104,'seconds':.01}
            def mask(data_root,record):
                result=np.zeros((4,4),dtype=np.uint8)
                if record['label']:result[1:3,1:3]=1
                return result
            args=SimpleNamespace(baseline_dir=baseline,data_root=data,warmup=1,repeats=1)
            with patch.object(runner,'load_checkpoint',side_effect=loader), patch.object(runner,'build_512',side_effect=build), \
                 patch.object(runner,'make_transform',return_value=None), \
                 patch.object(runner,'load_tensor',side_effect=lambda d,p,t:torch.full((3,4,4),values[p])), \
                 patch.object(runner,'load_mask',side_effect=mask):
                result=runner.run_experiment(args,(identities,original,manifest,cal,old,info),temp/'output')
            self.assertEqual(result['development_images'],2)
            self.assertEqual(result['held_out_images_opened'],0)
            self.assertIsNone(result['automatically_selected_resolution'])
            self.assertTrue(all(result['baseline_reproduction'].values()))
            self.assertEqual(result['config']['status'],'completed')
            for size in ('256','512'):
                perf=result['performance'][size]
                self.assertEqual(perf['samples'],2)
                self.assertGreater(perf['mean_ms'],0)
                self.assertGreater(perf['max_peak_allocated_bytes'],0)
                self.assertEqual(len(list((temp/f'output/{size}/maps/development').rglob('*.npz'))),2)
                self.assertEqual(result['qualities'][size]['area_groups']['lt_0.1_percent']['n'],0)
            self.assertEqual(json.loads((temp/'output/status.json').read_text())['status'],'completed')
            self.assertTrue((temp/'output/512/model.pt').exists())
            self.assertEqual(len(list((temp/'output/可视化').rglob('*.png'))),1)
            self.assertTrue((temp/'output/结果摘要.txt').exists())
            self.assertEqual(file_sha256(baseline/'model.pt'),identities['model.pt'])


if __name__=='__main__':
    unittest.main()
