"""Full-state persistence must preserve updates, RNG, validation and safe rollback."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

import train_actor_critic as trainer
import run_combat_move_pilot as shared
import run_reward_pilot as runner
import training_state as persistence

ROOT = Path(__file__).resolve().parent


class StateChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def same(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a.cpu(), b.cpu()))
        elif isinstance(a, np.ndarray):
            self.assertTrue(np.array_equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.same(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.same(x, y)
        else:
            self.assertEqual(a, b)

    def execute(self, folder, device='cpu', stop_after=None, resume='', save=True, mid_episode=False,
                mid_validation=False, fn=trainer.train):
        rows, actions, vals, stop = [], [], [], threading.Event()
        def episode(row):
            rows.append({**row, 'random_probes': (random.random(), np.random.random(), torch.rand((), device=device).item())})
            if not mid_episode and not mid_validation and row['episode'] == stop_after:
                stop.set()
        def step(env, ep, aid, act, opp):
            actions.append((ep, env.step_count, aid, act, env.grid.tolist(),
                            [(a.row, a.col, a.alive) for a in env.agents]))
            if mid_episode and ep == stop_after and env.step_count == 2:
                stop.set()
        def validation(row):
            if 'result' in row:
                vals.append(deepcopy(row))
            if mid_validation and row['episode'] == stop_after and row.get('completed') == 1:
                stop.set()
        kwargs = {} if fn is not trainer.train else {'save_training_state': save, 'resume_training_state': str(resume)}
        with patch('builtins.print'):
            _, rewards, wins = fn(phase='C', episodes=6, board_size=9, num_agents=2, max_steps=20,
                lr=1e-4, seed=123, device=device, checkpoint_dir=str(folder), resume_from=str(shared.WARM_START),
                validation_every=2, validation_episodes=2, validation_seed=21000, run_report=False,
                reward_diagnostics=True, advantage_diagnostics=True, log_every=2,
                on_step=step, on_episode=episode, on_validation=validation, stop_event=stop, **kwargs)
        return rows, actions, vals, rewards, wins

    def continuation(self, device='cpu', mid_episode=False, mid_validation=False):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='resume-test-') as name:
            base = Path(name)
            random.seed(17)
            full = self.execute(base / 'full', device)
            full_state = persistence.load_training_state(base / 'full/training_state.pt')
            probes = (random.random(), np.random.random(), torch.rand((), device=device).item())
            random.seed(17)
            cut = 3 if not mid_validation else 2
            partial = self.execute(base / 'split', device, stop_after=cut,
                                   mid_episode=mid_episode, mid_validation=mid_validation)
            checkpoint = base / 'split/training_state.pt'
            state = persistence.load_training_state(checkpoint)
            expected = cut - 1 if mid_episode or mid_validation else cut
            self.assertEqual(state['episode'], expected)
            self.assertEqual(state['scheduler']['last_epoch'], expected)
            self.assertEqual(state['scheduler']['T_max'], 6)
            self.assertTrue(all(v['step'].item() == expected for v in state['optimizer']['state'].values()))
            random.random(); np.random.random(); torch.rand(5, device=device)
            rest = self.execute(base / 'split', device, resume=checkpoint)
            resumed = persistence.load_training_state(checkpoint)
            for key in ('model', 'optimizer', 'scheduler', 'rng', 'histories', 'selection'):
                self.same(full_state[key], resumed[key])
            self.same(full[0], partial[0] + rest[0])
            retained_actions = [r for r in partial[1] if r[0] <= expected]
            self.same(full[1], retained_actions + rest[1])
            self.same(full[2], partial[2] + rest[2])
            self.same(probes, (random.random(), np.random.random(), torch.rand((), device=device).item()))
            self.same(full[3:], rest[3:])

    def test_cpu_continuation_matches_every_action_update_and_rng(self):
        self.continuation()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_deterministic_continuation_matches_every_action_update_and_rng(self):
        self.continuation('cuda')

    def test_partial_episode_is_discarded_and_replayed_exactly(self):
        self.continuation(mid_episode=True)

    def test_interrupted_validation_rolls_back_to_last_committed_episode(self):
        self.continuation(mid_validation=True)

    def test_optional_state_saves_preserve_default_and_before_source_training(self):
        spec = importlib.util.spec_from_file_location('trainer_before_resume',
            ROOT / 'runs/resume-implementation-20261015/before/train_actor_critic.py')
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='resume-test-') as name:
            results = []
            for index, (fn, save) in enumerate(((old.train, False), (trainer.train, False), (trainer.train, True))):
                random.seed(17)
                folder = Path(name) / str(index)
                results.append((self.execute(folder, fn=fn, save=save),
                    torch.load(folder / 'final_phaseC.pt', map_location='cpu', weights_only=True)))
            self.same(results[0], results[1])
            self.same(results[0], results[2])

    def test_atomic_replace_failure_keeps_previous_checkpoint_and_weights_are_not_states(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='resume-test-') as name:
            path = Path(name) / 'checkpoint.pt'
            persistence.atomic_save({'previous': 1}, path)
            before = path.read_bytes()
            with patch.object(persistence.os, 'replace', side_effect=OSError('simulated power loss')):
                with self.assertRaises(OSError):
                    persistence.atomic_save({'next': 2}, path)
            self.assertEqual(before, path.read_bytes())
            with self.assertRaises(ValueError):
                persistence.load_training_state(shared.WARM_START)

    def test_changed_config_source_runtime_or_phase_refuses_exact_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='resume-test-') as name:
            self.execute(Path(name), stop_after=1)
            path = Path(name) / 'training_state.pt'
            for key in ('config', 'sources', 'runtime'):
                state = persistence.load_training_state(path)
                state['provenance'][key]['modified'] = True
                altered = Path(name) / (key + '.pt')
                persistence.atomic_save(state, altered)
                with self.assertRaisesRegex(ValueError, 'exact resume refused'):
                    self.execute(Path(name), resume=altered)
            with self.assertRaises(ValueError):
                trainer.train(phase='D', save_training_state=True)

    def test_runner_restores_committed_logs_and_gzip_from_state_after_torn_writes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='resume-arm-test-') as name:
            base = Path(name)
            net = trainer.ActorCriticNet(board_size=9)
            trainer.smart_load_weights(net, str(shared.WARM_START), 'cpu')
            records = [{'group': group, 'steps': 1, 'kills': 1, 'owned_kill_success': True,
                **{key: False for key in ('suicide', 'two_or_more_placements', 'has_genuine_combo_pair',
                   'passive_win', 'actual_draw', 'skill_horizon_truncated')}} for group in ('escape', 'approach')]
            baseline = {'model_state_sha256': shared.model_digest(net), 'records': records}
            def probe(model, *args, **kwargs):
                return {'model_state_sha256': shared.model_digest(model), 'records': records}
            args = dict(attack_teacher='local', save_training_state=True, reward_diagnostics=True,
                        contact_diagnostics=True, normalize_advantage=False, smoke_episodes=4, smoke_steps=8)
            real_train = trainer.train
            stop = threading.Event()
            def cut_train(**kwargs):
                original = kwargs['on_episode']
                def episode(row):
                    original(row)
                    if row['episode'] == 2:
                        stop.set()
                kwargs['on_episode'] = episode
                return real_train(**kwargs)
            with patch.object(shared, 'skill_probe', side_effect=probe), patch('builtins.print'):
                with patch.object(trainer, 'train', side_effect=cut_train):
                    shared.train_arm(base, 'control', 'cpu', True, stop, baseline, **args)
                for name in ('metrics.jsonl', 'skill_metrics.jsonl', 'skill_games.jsonl.gz',
                             'validation_metrics.jsonl', 'best_validation.json', 'best_phaseC.pt'):
                    (base / 'control' / name).write_bytes(b'torn write')
                resumed = shared.train_arm(base, 'control', 'cpu', True, threading.Event(), baseline, resume=True, **args)
            self.assertTrue(shared.pair_ready({'control': resumed}, ('control',)))
            self.assertEqual(resumed['training_state']['episode'], 4)
            rows = [json.loads(line) for line in (base / 'control/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['optimizer_steps'] for r in rows], [1, 2, 3, 4])
            vals = [json.loads(line) for line in (base / 'control/validation_metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['episode'] for r in vals], [0, 2, 4])

    def test_continuation_in_a_new_python_process_cpu_and_cuda(self):
        program = '''import json,os,random,sys,threading
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import numpy as np
import torch
import train_actor_critic as trainer
from run_combat_move_pilot import WARM_START
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
folder,mode,device=Path(sys.argv[2]),sys.argv[3],sys.argv[4]
folder.mkdir(exist_ok=True)
random.seed(17)
stop=threading.Event()
with (folder/'events.jsonl').open('a' if mode=='resume' else 'w') as handle:
    def step(env,ep,aid,actions,opp):
        handle.write(json.dumps(['step',ep,env.step_count,aid,actions,env.grid.tolist()])+'\\n')
    def episode(row):
        handle.write(json.dumps(['episode',row,random.random(),float(np.random.random()),torch.rand((),device=device).item()])+'\\n')
        handle.flush()
        if mode=='pause' and row['episode']==2: stop.set()
    trainer.train(phase='C',episodes=4,board_size=9,num_agents=2,max_steps=12,lr=1e-4,seed=123,device=device,
        checkpoint_dir=str(folder),resume_from=str(WARM_START),save_training_state=True,
        resume_training_state=str(folder/'training_state.pt') if mode=='resume' else '',
        validation_every=2,validation_episodes=2,validation_seed=21000,log_every=2,run_report=False,
        on_step=step,on_episode=episode,stop_event=stop)
'''
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='resume-process-test-') as name:
            base = Path(name)
            script = base / 'child.py'
            script.write_text(program, encoding='utf-8')
            for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
                full, split = base / (device + '-full'), base / (device + '-split')
                for directory, mode in ((full, 'full'), (split, 'pause'), (split, 'resume')):
                    result = subprocess.run([sys.executable, str(script), str(ROOT), str(directory), mode, device],
                        cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((full / 'events.jsonl').read_bytes(), (split / 'events.jsonl').read_bytes())
                a = persistence.load_training_state(full / 'training_state.pt')
                b = persistence.load_training_state(split / 'training_state.pt')
                for key in ('model', 'optimizer', 'scheduler', 'rng', 'histories', 'selection'):
                    self.same(a[key], b[key])

    def test_completed_evaluation_is_reused_only_for_the_locked_checkpoint(self):
        from test_combat_move_pilot import result
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='resume-eval-test-') as name:
            folder = Path(name)
            cached = {**result(seed=36000), 'max_steps': 500, 'checkpoint_sha256': 'locked'}
            path = folder / 'evaluation_original_36000.json'
            path.write_text(json.dumps(cached))
            with patch.object(shared, 'digest', return_value='locked'), \
                 patch.object(shared, 'make_ac_fn', return_value=object()) as load, \
                 patch.object(shared, 'evaluate_seeded', side_effect=lambda *a, **k: result(seed=a[4])):
                results = shared.evaluate_models(folder, threading.Event(), False, arm_names=runner.ARMS,
                                                  test_seed=36000, resume=True)
                self.assertEqual(results['original'], cached)
                self.assertEqual(load.call_count, 2)
                cached['checkpoint_sha256'] = 'different'
                path.write_text(json.dumps(cached))
                with self.assertRaisesRegex(ValueError, 'locked model'):
                    shared.evaluate_models(folder, threading.Event(), False, arm_names=runner.ARMS,
                                           test_seed=36000, resume=True)

    def test_old_format_and_expired_deadline_refuse_before_log_recovery(self):
        from datetime import datetime, timezone, timedelta
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='resume-budget-test-') as name:
            folder = Path(name)
            path = folder / 'config.json'
            path.write_text(json.dumps({'resumable_training_state': False}))
            with self.assertRaisesRegex(ValueError, 'Resume config'):
                runner.run(folder, smoke=True, resume=True)
            stored = {'resumable_training_state': True,
                'registered_utc': (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
                'source_sha256': {f: shared.digest(ROOT / f) for f in runner.SOURCE_FILES},
                'preflight_input_sha256': {str(p): shared.digest(p) for p in runner.PREFLIGHT_INPUTS},
                'train_device': 'cpu', 'full_protocol': False, 'arms': runner.ARMS,
                'budget_minutes': {'overall': 150, 'preflight': 15, 'training': 100, 'evaluation': 50},
                'runtime': persistence.runtime_state()}
            path.write_text(json.dumps(stored))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'deadline expired'):
                runner.run(folder, smoke=True, resume=True)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((folder / 'resume_history').exists())


if __name__ == '__main__':
    unittest.main()
