"""D1 runner protocol guards; actual optimizer/state behavior lives in test_ppo_d1."""
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock

import torch

from audit_t2 import digest,dump
import run_ppo_pilot as runner

ROOT=Path(__file__).resolve().parent


class PilotChecks(unittest.TestCase):
    def test_refuses_overwrite_wrong_b_device_budget_and_missing_resume(self):
        with patch.object(Path,'mkdir') as mkdir:
            with self.assertRaises(FileExistsError): runner.run(ROOT)
            for budget in (0,151,float('nan')):
                with self.assertRaises(ValueError): runner.run(ROOT/'unused-d1',budget_minutes=budget)
            with self.assertRaises(ValueError): runner.run(ROOT/'unused-d1',resume=True)
            with patch.object(runner,'digest',return_value='wrong'):
                with self.assertRaises(ValueError): runner.run(ROOT/'unused-d1')
            with patch.object(torch.cuda,'is_available',return_value=False):
                with self.assertRaises(ValueError): runner.run(ROOT/'unused-d1',device='cuda')
            mkdir.assert_not_called()

    def fake_run(self,out,resume=False):
        calls=[]
        def arm(output,name,device,smoke,stop,baseline,**kwargs):
            calls.append(kwargs)
            folder=output/name
            folder.mkdir()
            checkpoints={}
            for kind in ('best','final'):
                path=folder/f'{kind}_phaseC.pt'
                path.write_bytes(name.encode())
                checkpoints[kind]={'path':str(path),'sha256':digest(path)}
            state=folder/'training_state.pt'
            state.write_bytes(b'fake-state')
            result=dict(training_complete=True,validation_complete=True,skill_probes_complete=True,
                completed_episodes=20,expected_episodes=20,optimizer_instances=1,optimizer_steps=80,
                checkpoints=checkpoints,training_state={'path':str(state),'sha256':digest(state),'episode':20})
            dump(folder/'summary.json',result)
            return result
        def evaluate(*args,**kwargs):
            return {name:{'games':[],'num_episodes':2} for name in ('original','control','ppo')}
        with patch.object(runner,'BaseNet',return_value=MagicMock()), \
             patch('train_actor_critic.smart_load_weights'),patch.object(runner,'skill_probe',return_value={}), \
             patch.object(runner,'probe_summary',side_effect=lambda x:x), \
             patch.object(runner,'train_arm',side_effect=arm),patch.object(runner,'verify_updates'), \
             patch.object(runner,'evaluate_models',side_effect=evaluate),patch('builtins.print'):
            result=runner.run(out,smoke=True,resume=resume)
        return result,calls

    def test_smoke_wires_only_objective_difference_and_never_qualifies_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs',prefix='d1-runner-') as name:
            out=Path(name)/'run'
            summary,calls=self.fake_run(out)
            self.assertEqual(summary['status'],'complete')
            self.assertFalse(summary['full_protocol'] or summary['eligible_for_replication'])
            self.assertEqual(summary['decision'],'retain_original')
            self.assertEqual([c['policy_objective'] for c in calls],['ratio','ppo'])
            self.assertEqual({k:v for k,v in calls[0].items() if k!='policy_objective'},
                             {k:v for k,v in calls[1].items() if k!='policy_objective'})
            for call in calls:
                self.assertEqual((call['update_epochs'],call['clip_epsilon'],call['target_kl']),(4,.2,.02))
                self.assertTrue(call['save_training_state'])
                self.assertEqual(call['attack_teacher'],'local')
            cfg=json.loads((out/'config.json').read_text())
            self.assertEqual(cfg['evaluation']['seed'],39870)
            self.assertEqual(json.loads((out/'selected_models.json').read_text())['original']['sha256'],runner.B_SHA256)

    def test_resume_rejects_source_changes_expiry_and_reselection(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs',prefix='d1-resume-runner-') as name:
            out=Path(name)/'run'
            self.fake_run(out)
            summary=json.loads((out/'summary.json').read_text())
            summary['status']='incomplete'
            dump(out/'summary.json',summary)
            cfg=json.loads((out/'config.json').read_text())
            bad=dict(cfg)
            bad['target_kl']=.04
            dump(out/'config.json',bad)
            with self.assertRaisesRegex(ValueError,'Resume source/config/runtime changed'): self.fake_run(out,True)
            bad={**cfg,'registered_utc':(datetime.now(timezone.utc)-timedelta(hours=3)).isoformat()}
            dump(out/'config.json',bad)
            with self.assertRaisesRegex(ValueError,'deadline expired'): self.fake_run(out,True)
            dump(out/'config.json',cfg)
            selected=json.loads((out/'selected_models.json').read_text())
            selected['ppo']['sha256']='wrong'
            dump(out/'selected_models.json',selected)
            with self.assertRaisesRegex(ValueError,'reselection forbidden'): self.fake_run(out,True)

    def test_verify_updates_rejects_optimizer_count_drift(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs',prefix='d1-counts-') as name:
            out=Path(name)
            arms={}
            states={}
            for arm,objective in runner.ARMS.items():
                folder=out/arm
                folder.mkdir()
                epoch={'epoch':1,'exact_kl_after_update':.05}
                row={'optimizer_steps':1,'ratio_update':{'objective':objective,'epochs_requested':4,'epochs':1,
                    'clipped':objective=='ppo','target_kl':.02,'clip_epsilon':.2,'epoch_metrics':[epoch],
                    'guard_exceeded':True,'early_stop':True}}
                (folder/'metrics.jsonl').write_text(json.dumps(row)+'\n')
                arms[arm]={'completed_episodes':1,'optimizer_steps':1}
                states[arm]={'episode':1,'optimizer':{'state':{0:{'step':torch.tensor(1.)}}},'scheduler':{'last_epoch':1}}
            with patch.object(runner,'load_training_state',side_effect=lambda path:states[path.parent.name]):
                runner.verify_updates(out,arms)
                arms['ppo']['optimizer_steps']=2
                with self.assertRaises(AssertionError): runner.verify_updates(out,arms)


if __name__=='__main__': unittest.main()
