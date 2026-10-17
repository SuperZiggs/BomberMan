"""D1 clipping, fixed-target reuse, KL guards, cancellation and exact continuation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
from types import MethodType
import unittest
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

import train_actor_critic as trainer
import run_combat_move_pilot as shared
from test_ppo_d0 import TinyNet, tape
from training_state import load_training_state

ROOT = Path(__file__).resolve().parent
B = ROOT/'runs/pressure-C-next/final_phaseC.pt'


def same(test, a, b, path='state'):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a.cpu(), b.cpu()), path)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a: same(test,a[k],b[k],path+'.'+str(k))
    elif isinstance(a, (list,tuple)):
        test.assertEqual(len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)): same(test,x,y,path+'.'+str(i))
    else: test.assertEqual(a,b,path)


def update(net, opt, data, *, clipped=False, epochs=4, target_kl=.02, stop=None):
    return trainer.update_ratio_rollout(net,opt,data,torch.arange(1,len(data)+1,dtype=torch.float64),
        torch.ones(len(data),dtype=torch.float64),epochs=epochs,clipped=clipped,clip_epsilon=.2,
        target_kl=target_kl,value_coef=.5,entropy_coef=.015,stop_event=stop)


class D1MathChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_clipped_surrogate_signs_and_gradient_plateaus_by_hand(self):
        for ratio in (.5,2.5):
            for advantage in (-2.,3.):
                net = TinyNet()
                with torch.no_grad():
                    net.actor.weight.zero_()
                    net.actor.bias.zero_()
                obs = torch.zeros(1,1,2,dtype=torch.float64)
                mask = torch.tensor([True,True,True,False,True,True])
                old = torch.tensor([.2,.2,.2,0.,.2,.2],dtype=torch.float64)
                step = trainer.freeze_rollout_step(obs,0,mask,old[0].log(),torch.tensor(0.,dtype=torch.float64),
                    old,0.,True,'match_end')
                probability = .2*ratio
                with torch.no_grad():
                    net.actor.bias[:] = torch.log(torch.full((6,),(1-probability)/4,dtype=torch.float64))
                    net.actor.bias[0] = torch.log(torch.tensor(probability))
                result = trainer.ratio_rollout_losses(net,[step],torch.tensor([advantage],dtype=torch.float64),
                    torch.zeros(1,dtype=torch.float64),clip_epsilon=.2)
                actual_ratio = result['ratios'][0].item()
                expected = -min(actual_ratio*advantage,max(.8,min(1.2,actual_ratio))*advantage)
                self.assertAlmostEqual(result['policy_loss'].item(),expected,places=12)
                result['policy_loss'].backward()
                plateau = (ratio > 1.2 and advantage > 0) or (ratio < .8 and advantage < 0)
                self.assertEqual(net.actor.bias.grad.abs().max().item() == 0.,plateau)

    def test_four_epochs_keep_old_policy_masks_targets_and_group_weights_fixed(self):
        for clipped in (False,True):
            torch.manual_seed(7)
            net = TinyNet()
            data = tape(net)
            before = deepcopy(data)
            opt = torch.optim.Adam(net.parameters(),lr=1e-4)
            real = trainer.ratio_rollout_losses
            calls = []
            def replay(*args,**kwargs):
                calls.append((args[2].detach().clone(),args[3].detach().clone()))
                return real(*args,**kwargs)
            with patch.object(trainer,'ratio_rollout_losses',side_effect=replay), \
                 patch.object(trainer,'get_action_mask',side_effect=AssertionError('recomputed mask')), \
                 patch.object(trainer,'get_attack_action',side_effect=AssertionError('new label')):
                result = update(net,opt,data,clipped=clipped)
            self.assertEqual(result['diagnostics']['epochs'],4)
            self.assertEqual(len(calls),8)
            self.assertFalse(result['diagnostics']['early_stop'])
            self.assertEqual({int(s['step']) for s in opt.state.values()},{4})
            for a,b in calls: same(self,a,calls[0][0]); same(self,b,calls[0][1])
            same(self,data,before)
            for row in result['diagnostics']['epoch_metrics']:
                self.assertEqual(row['clipped'],clipped)
                self.assertEqual(row['objective'],'ppo' if clipped else 'ratio')
                self.assertEqual(row['label_counts'],dict(escape=1,combat=1,navigation=2))
                self.assertTrue(all(k in row for k in ('ratio_p95','clip_fraction','entropy','value_loss','exact_kl_after_update')))

    def test_same_post_update_kl_guard_stops_remaining_epochs_in_both_arms(self):
        for clipped in (False,True):
            net = TinyNet()
            data = tape(net)
            opt = torch.optim.Adam(net.parameters(),lr=.5)
            result = update(net,opt,data,clipped=clipped)
            d = result['diagnostics']
            self.assertEqual(d['epochs'],1)
            self.assertTrue(d['early_stop'] and d['guard_exceeded'])
            self.assertGreater(d['epoch_metrics'][0]['exact_kl_after_update'],.03)
            self.assertEqual({int(s['step']) for s in opt.state.values()},{1})

    def test_disabled_guard_runs_all_epochs_even_when_kl_is_large(self):
        net = TinyNet()
        data = tape(net)
        result = update(net,torch.optim.Adam(net.parameters(),lr=.2),data,target_kl=0)
        self.assertEqual(result['diagnostics']['epochs'],4)
        self.assertFalse(result['diagnostics']['guard_exceeded'])

    def test_cancellation_after_second_step_restores_existing_adam_and_parameters(self):
        net = TinyNet()
        data = tape(net)
        opt = torch.optim.Adam(net.parameters(),lr=1e-4)
        update(net,opt,data,epochs=1)
        before = deepcopy((net.state_dict(),opt.state_dict()))
        stop = threading.Event()
        real, calls = opt.step, []
        def step(*a,**k):
            real(*a,**k)
            calls.append(1)
            if len(calls) == 2: stop.set()
        with patch.object(opt,'step',side_effect=step):
            self.assertIsNone(update(net,opt,data,stop=stop))
        same(self,before,(net.state_dict(),opt.state_dict()))
        self.assertTrue(all(p.grad is None for p in net.parameters()))

    def test_nonfinite_gradient_stops_before_adam(self):
        net = TinyNet()
        data = tape(net)
        opt = torch.optim.Adam(net.parameters(),lr=1e-4)
        handle = net.actor.bias.register_hook(lambda g: torch.full_like(g,float('inf')))
        with self.assertRaises(FloatingPointError): update(net,opt,data)
        self.assertFalse(opt.state)
        handle.remove()

    def test_bad_epoch_clip_kl_and_ac_combinations_rejected_before_env(self):
        with patch.object(trainer,'BombermanFFAEnv') as env:
            for kwargs in ({'update_epochs':0},{'update_epochs':True},{'clip_epsilon':float('nan')},
                           {'target_kl':float('inf')},{'target_kl':-.1},{'update_epochs':4},
                           {'target_kl':.02}):
                with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): trainer.train(**kwargs)
            env.assert_not_called()


class D1TrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False

    def execute(self,folder,objective='ratio',device='cpu',pause=False,resume=False,cancel_step=None):
        rows,stop = [],threading.Event()
        real_adam = torch.optim.Adam
        updates=[]
        def adam(*a,**k):
            opt=real_adam(*a,**k)
            real_step=opt.step
            def step(_opt,*args,**kwargs):
                result=real_step(*args,**kwargs)
                updates.append(1)
                if cancel_step and len(updates)==cancel_step: stop.set()
                return result
            opt.step=MethodType(step,opt)
            return opt
        def record(row):
            rows.append(row)
            if pause and row['episode']==1: stop.set()
        random.seed(17)
        with patch.object(trainer.optim,'Adam',side_effect=adam),patch('builtins.print'):
            trainer.train(episodes=3,board_size=9,max_steps=8,seed=33,device=device,lr=1e-4,
                checkpoint_dir=str(folder),resume_from=str(B),policy_objective=objective,update_epochs=4,
                target_kl=.02,save_training_state=True,validation_every=0,run_report=False,
                on_episode=record,stop_event=stop,
                resume_training_state=str(folder/'training_state.pt') if resume else '')
        return load_training_state(folder/'training_state.pt'),rows

    def test_four_epoch_training_resume_and_mid_epoch_cancel_cpu_cuda(self):
        for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
            for objective in ('ratio','ppo'):
                with tempfile.TemporaryDirectory(dir=ROOT/'runs',prefix='d1-state-') as name:
                    base=Path(name)
                    full,rows=self.execute(base/'full',objective,device)
                    partial,_=self.execute(base/'paused',objective,device,cancel_step=6)
                    self.assertEqual(partial['episode'],1)
                    self.assertEqual({int(v['step']) for v in partial['optimizer']['state'].values()},{4})
                    resumed,_=self.execute(base/'paused',objective,device,resume=True)
                    same(self,full,resumed)
                    self.assertEqual(sum(r['ratio_update']['epochs'] for r in rows),int(next(iter(full['optimizer']['state'].values()))['step']))

    def test_one_epoch_ppo_and_ratio_are_identical_at_old_policy(self):
        torch.manual_seed(4)
        first=TinyNet()
        second=deepcopy(first)
        data=tape(first)
        for net,clipped in ((first,False),(second,True)):
            update(net,torch.optim.Adam(net.parameters(),lr=1e-4),data,epochs=1,clipped=clipped)
        same(self,first.state_dict(),second.state_dict())

    def test_new_python_process_ppo_resume_cpu_cuda(self):
        program='''import json,os,random,sys,threading
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
import train_actor_critic as trainer
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False
folder,mode,device=Path(sys.argv[2]),sys.argv[3],sys.argv[4]
folder.mkdir(exist_ok=True)
stop=threading.Event()
random.seed(17)
with (folder/'events.jsonl').open('a' if mode=='resume' else 'w') as f:
    def frame(env,ep,aid,actions,opponents):
        f.write(json.dumps(['step',ep,aid,actions,env.grid.tolist()])+'\\n')
    def row(r):
        f.write(json.dumps(['episode',r])+'\\n')
        f.flush()
        if mode=='pause' and r['episode']==1:stop.set()
    trainer.train(episodes=3,board_size=9,max_steps=8,seed=33,device=device,lr=1e-4,
        checkpoint_dir=str(folder),resume_from=str(Path(sys.argv[1])/'runs/pressure-C-next/final_phaseC.pt'),
        policy_objective='ppo',update_epochs=4,target_kl=.02,save_training_state=True,
        validation_every=0,run_report=False,on_step=frame,on_episode=row,stop_event=stop,
        resume_training_state=str(folder/'training_state.pt') if mode=='resume' else '')
'''
        for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
            with tempfile.TemporaryDirectory(dir=ROOT/'runs',prefix='d1-process-') as name:
                base=Path(name)
                for folder,mode in (('full','full'),('paused','pause'),('paused','resume')):
                    result=subprocess.run([sys.executable,'-c',program,str(ROOT),str(base/folder),mode,device],
                        capture_output=True,text=True,timeout=120)
                    self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual((base/'full/events.jsonl').read_bytes(),(base/'paused/events.jsonl').read_bytes())
                same(self,load_training_state(base/'full/training_state.pt'),load_training_state(base/'paused/training_state.pt'))

    def test_readiness_stays_strict_for_single_update_callers(self):
        arm=dict(training_complete=True,validation_complete=True,skill_probes_complete=True,
            completed_episodes=200,expected_episodes=200,optimizer_instances=1,optimizer_steps=800)
        self.assertFalse(shared.pair_ready({'x':arm},('x',)))
        self.assertTrue(shared.pair_ready({'x':arm},('x',),max_updates_per_episode=4))
        arm['optimizer_steps']=801
        self.assertFalse(shared.pair_ready({'x':arm},('x',),max_updates_per_episode=4))


if __name__=='__main__': unittest.main()
