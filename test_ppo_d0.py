"""D0 checks frozen data, masked ratio gradients, group means and one Adam update."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import random
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F

import train_actor_critic as trainer
from benchmark_t3 import observations
from test_combat import arena, put_bomb
from training_state import load_training_state
from legacy_test_support import legacy_train, strip_sd

ROOT = Path(__file__).resolve().parent
B = ROOT / 'runs/pressure-C-next/final_phaseC.pt'


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(2, 6, dtype=torch.float64)
        self.value = torch.nn.Linear(2, 1, dtype=torch.float64)

    def forward(self, x):
        return self.actor(x.flatten(1)), self.value(x.flatten(1)).squeeze(-1)


def tape(net, groups=('escape', 'navigation', 'navigation', 'combat', None)):
    steps = []
    for i, group in enumerate(groups):
        obs = torch.tensor([[[i / 10, -.2]]], dtype=torch.float64)
        mask = torch.tensor([True, i != 0, True, False, True, True])
        logits, value = net(obs.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))
        action = 0 if i == 0 else 1
        steps.append(trainer.freeze_rollout_step(obs, action, mask, dist.log_prob(torch.tensor([action])),
            value, dist.probs[0], float(i), i == len(groups)-1, 'time_limit' if i == len(groups)-1 else None,
            group, (0, 2) if group == 'combat' else (0,) if group is not None else ()))
    return steps


class RolloutChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_freeze_clones_detaches_and_does_not_consume_rng(self):
        torch.manual_seed(7)
        net = TinyNet()
        before, np_before = torch.get_rng_state().clone(), deepcopy(np.random.get_state())
        steps = tape(net)
        for step in steps:
            for value in step.values():
                if isinstance(value, torch.Tensor):
                    self.assertFalse(value.requires_grad)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        np.testing.assert_equal(np_before, np.random.get_state())
        obs = torch.zeros(1, 1, 2, requires_grad=True)
        probs = torch.tensor([.5, .5, 0., 0., 0., 0.], requires_grad=True)
        mask = probs.detach() > 0
        saved = trainer.freeze_rollout_step(obs, 0, mask, probs[0].log(), obs.sum(), probs,
                                          0., True, 'match_end', 'combat', (0, 1))
        with torch.no_grad():
            obs.add_(4)
            probs.fill_(1/6)
            mask[:] = True
        self.assertEqual(saved['obs'].sum().item(), 0)
        self.assertEqual(saved['mask'].sum().item(), 2)
        self.assertEqual(saved['old_probs'][0].item(), .5)
        self.assertEqual(saved['teacher_target'].tolist(), [.5, .5, 0., 0., 0., 0.])

    def test_ratio_one_group_means_and_gradient_adam_parity_float64(self):
        torch.manual_seed(9)
        base = TinyNet()
        data = tape(base)
        advantages = torch.tensor([1., -2., .5, 4., -.2], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64, requires_grad=True)
        ratio_net, ac_net = deepcopy(base), deepcopy(base)
        result = trainer.ratio_rollout_losses(ratio_net, data, advantages, targets)
        logits, value = ac_net(torch.stack([s['obs'] for s in data]))
        masks = torch.stack([s['mask'] for s in data])
        logits = logits.masked_fill(~masks, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        ref = -(dist.log_prob(torch.tensor([s['action'] for s in data])) * advantages.detach()).mean()
        ref += .5 * F.mse_loss(value, targets.detach()) - .015 * dist.entropy().mean()
        loss = result['policy_loss'] + .5*result['value_loss'] - .015*result['entropy']
        self.assertLess(result['diagnostics']['ratio_max_abs_error_from_one'], 1e-14)
        self.assertAlmostEqual(result['diagnostics']['exact_kl_before_update'], 0., delta=1e-14)
        self.assertEqual(result['diagnostics']['label_counts'], {'escape': 1, 'combat': 1, 'navigation': 2})
        for group, coefficient in (('escape', .5), ('combat', .5), ('navigation', .2)):
            indices = [i for i, s in enumerate(data) if s['teacher_group'] == group]
            expected = torch.stack([-F.log_softmax(logits[i], -1)[[0, 2]].mean() if group == 'combat'
                                    else F.cross_entropy(logits[i:i+1], torch.tensor([0])) for i in indices]).mean()
            torch.testing.assert_close(result[group+'_loss'], expected, rtol=1e-13, atol=1e-13)
            ref += coefficient * expected
            loss += coefficient * result[group+'_loss']
        loss.backward()
        ref.backward()
        self.assertIsNone(advantages.grad)
        self.assertIsNone(targets.grad)
        for a, b in zip(ratio_net.parameters(), ac_net.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
        for model in (ratio_net, ac_net):
            torch.optim.Adam(model.parameters(), lr=1e-4).step()
        for a, b in zip(ratio_net.parameters(), ac_net.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_one_allowed_action_has_ratio_one_zero_entropy_and_finite_gradient(self):
        net = TinyNet()
        step = tape(net, (None,))[0]
        mask = torch.tensor([True, False, False, False, False, False])
        step = trainer.freeze_rollout_step(step['obs'], 0, mask, torch.tensor(0., dtype=torch.float64),
            step['old_value'], mask.to(torch.float64), 1., True, 'match_end')
        result = trainer.ratio_rollout_losses(net, [step], torch.ones(1), torch.ones(1, dtype=torch.float64))
        self.assertEqual(result['entropy'].item(), 0.)
        self.assertEqual(result['diagnostics']['ratio_mean'], 1.)
        self.assertEqual(result['diagnostics']['label_counts'], dict(escape=0, combat=0, navigation=0))
        (result['policy_loss'] + result['value_loss']).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_invalid_action_old_probabilities_labels_and_boundaries_rejected(self):
        step = tape(TinyNet(), (None,))[0]
        args = dict(obs=step['obs'], action=step['action'], mask=step['mask'], old_log_prob=step['old_log_prob'],
            old_value=step['old_value'], old_probs=step['old_probs'], reward=0., terminal=True, boundary='match_end')
        for change in ({'action': 3}, {'action': -1}, {'action': True}, {'mask': torch.zeros(6, dtype=torch.bool)},
                       {'old_log_prob': torch.tensor(0.)}, {'old_probs': torch.ones(6)/6},
                       {'old_value': torch.tensor(float('nan'))}, {'reward': float('inf')},
                       {'terminal': False}, {'boundary': None}, {'teacher_group': 'bad'},
                       {'teacher_group': 'combat', 'teacher_actions': (3,)},
                       {'teacher_group': 'combat', 'teacher_actions': (0, 0)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                trainer.freeze_rollout_step(**{**args, **change})

    def test_replay_uses_stored_masks_labels_old_data_and_positive_kl_after_change(self):
        net = TinyNet()
        data = tape(net)
        before = deepcopy(data)
        for p in net.parameters():
            with torch.no_grad():
                p.add_(.1 * torch.arange(p.numel()).reshape(p.shape))
        with patch.object(trainer, 'get_action_mask', side_effect=AssertionError('Mask recomputed')), \
             patch.object(trainer, 'get_attack_action', side_effect=AssertionError('Teacher called')):
            result = trainer.ratio_rollout_losses(net, data, torch.ones(5), torch.zeros(5, dtype=torch.float64))
        self.assertGreater(result['diagnostics']['exact_kl_before_update'], 0)
        self.assertGreater(result['diagnostics']['ratio_max_abs_error_from_one'], 0)
        for x, y in zip(data, before):
            for key in x:
                self.assertTrue(torch.equal(x[key], y[key])) if isinstance(x[key], torch.Tensor) else self.assertEqual(x[key], y[key])
        with self.assertRaises(ValueError):
            trainer.ratio_rollout_losses(net, [], torch.ones(0), torch.ones(0))
        data[0]['terminal'] = True
        with self.assertRaises(ValueError):
            trainer.ratio_rollout_losses(net, data, torch.ones(5), torch.zeros(5))


class D0TrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    def observe(self, objective='ac', device='cpu', max_steps=20, env=None, stop=None, hook=None, fn=trainer.train):
        rows, actions, gradients, optimizers, replays = [], [], [], [], []
        real_clip, real_adam, real_replay = torch.nn.utils.clip_grad_norm_, torch.optim.Adam, trainer.ratio_rollout_losses
        def clip(parameters, *args, **kwargs):
            parameters = list(parameters)
            gradients.append([p.grad.detach().clone() if p.grad is not None else None for p in parameters])
            return real_clip(parameters, *args, **kwargs)
        def adam(*args, **kwargs):
            opt = real_adam(*args, **kwargs)
            optimizers.append(opt)
            return opt
        def replay(*args, **kwargs):
            with patch.object(trainer, 'get_action_mask', side_effect=AssertionError('Mask queried during update')), \
                 patch.object(trainer, 'get_attack_action', side_effect=AssertionError('Teacher queried during update')), \
                 patch.object(trainer, 'get_two_bomb_plan', side_effect=AssertionError('Planner queried during update')):
                result = real_replay(*args, **kwargs)
            replays.append(args[1])
            return result
        def frame(current, ep, aid, sampled, opponents):
            if current.step_count:
                actions.append((aid, list(sampled)))
            if hook:
                hook(current, ep, aid, sampled, opponents)
        with patch('torch.save'), patch('os.makedirs'), patch('builtins.print'), \
             patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=clip), \
             patch.object(trainer.optim, 'Adam', side_effect=adam), \
             patch.object(trainer, 'ratio_rollout_losses', side_effect=replay):
            kwargs = dict(episodes=1, board_size=9 if env is None else env.board_size,
                num_agents=2 if env is None else env.num_agents, max_steps=max_steps, lr=1e-4,
                device=device, seed=2026101506, resume_from=str(B) if env is None else '',
                checkpoint_dir='unused-d0-unit', run_report=False, validation_every=0,
                advantage_diagnostics=True, stop_event=stop, on_step=frame, on_episode=rows.append)
            if fn is trainer.train:
                kwargs['policy_objective'] = objective
            if env is None:
                net, rewards, wins = fn(**kwargs)
            else:
                with patch.object(trainer, 'BombermanFFAEnv', return_value=env), \
                     patch.object(env, 'reset', side_effect=lambda: observations(env)), \
                     patch.object(trainer.RuleBasedOpponent, 'select_action_env', return_value=0), \
                     patch('torch.distributions.Categorical.sample', side_effect=lambda: torch.tensor([0], device=device)):
                    net, rewards, wins = fn(**kwargs)
        return {'state': deepcopy(net.state_dict()), 'rows': rows, 'actions': actions, 'gradients': gradients,
                'optimizer': deepcopy(optimizers[0].state_dict()), 'replays': replays, 'rewards': rewards, 'wins': wins}

    def test_real_cnn_first_trajectory_losses_gradients_and_adam_cpu_cuda(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            ac, ratio = self.observe(device=device), self.observe('ratio', device=device)
            self.assertEqual(ac['actions'], ratio['actions'])
            self.assertEqual(ac['rewards'], ratio['rewards'])
            self.assertEqual(ac['wins'], ratio['wins'])
            for field in ('value_loss', 'escape_loss', 'combat_loss', 'navigation_loss'):
                self.assertAlmostEqual(ac['rows'][0][field], ratio['rows'][0][field], delta=2e-5)
            self.assertNotAlmostEqual(ac['rows'][0]['policy_loss'], ratio['rows'][0]['policy_loss'], delta=1e-4)
            for a, b in zip(ac['gradients'][0], ratio['gradients'][0]):
                self.assertIsNone(b) if a is None else torch.testing.assert_close(a, b, rtol=2e-4, atol=3e-6)
            for key, value in ac['state'].items():
                # Batched float32 reductions differ; Adam amplifies rounding near zero gradients.
                torch.testing.assert_close(value, ratio['state'][key], rtol=1e-4, atol=2e-5)
            for key, state in ratio['optimizer']['state'].items():
                self.assertEqual(state['step'].item(), 1)
                for field in ('exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(ac['optimizer']['state'][key][field], state[field], rtol=2e-4, atol=3e-7)
            metrics = ratio['rows'][0]['ratio_update']
            self.assertLess(metrics['ratio_max_abs_error_from_one'], 1e-4)
            self.assertEqual(metrics['samples'], ratio['rows'][0]['learner_steps'])
            self.assertEqual(metrics['last_boundary'], 'time_limit')
            for sampled, stored in zip(ratio['actions'], ratio['replays'][0]):
                self.assertEqual(sampled[1][sampled[0]], stored['action'])
                self.assertTrue(stored['mask'][stored['action']])

    def test_default_ac_matches_pre_d0_actions_metrics_and_weights_exactly(self):
        path = ROOT / 'runs/ppo-d0-implementation-20261015/before/train_actor_critic.py'
        spec = importlib.util.spec_from_file_location('before_d0_trainer', path)
        before = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(before)
        old, new = self.observe(fn=before.train), self.observe(fn=legacy_train)
        for key in ('actions', 'rows', 'rewards', 'wins'):
            self.assertEqual(strip_sd(old[key]), strip_sd(new[key]))
        for key in old['state']:
            self.assertTrue(torch.equal(old['state'][key], new['state'][key]))

    def test_learner_death_ends_rollout_before_ffa_bookkeeping(self):
        env = arena(3)
        env.max_steps = 4
        put_bomb(env, 1, 1, 1, timer=1, blast_range=2)
        result = self.observe('ratio', env=env, max_steps=4)
        self.assertEqual(len(result['actions']), 4)
        self.assertEqual(len(result['replays'][0]), 1)
        self.assertEqual(result['replays'][0][-1]['boundary'], 'learner_death')

    def test_delayed_owned_kill_and_win_records_terminal_reward(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 1, 5
        put_bomb(env, 1, 3, 0, timer=3, blast_range=2)
        result = self.observe('ratio', env=env, max_steps=10)
        self.assertEqual(len(result['replays'][0]), 3)
        self.assertEqual(result['replays'][0][-1]['boundary'], 'match_end')
        self.assertGreater(result['replays'][0][-1]['reward'], 10)
        self.assertEqual(result['rows'][0]['active_win'], 1)

    def test_timeout_500_is_one_complete_terminal_trajectory(self):
        env = arena(2)
        env.max_steps = 500
        result = self.observe('ratio', env=env, max_steps=500)
        steps = result['replays'][0]
        self.assertEqual(len(steps), 500)
        self.assertEqual(sum(s['terminal'] for s in steps), 1)
        self.assertEqual(steps[-1]['boundary'], 'time_limit')
        self.assertTrue(all(a.alive for a in env.agents))

    def test_cancelled_partial_rollout_makes_no_update(self):
        stop = threading.Event()
        result = self.observe('ratio', stop=stop, hook=lambda env, *a: stop.set() if env.step_count == 2 else None)
        self.assertFalse(result['rows'])
        self.assertFalse(result['replays'])
        self.assertFalse(result['gradients'])
        self.assertFalse(result['optimizer']['state'])

    def test_bad_objective_rejected_before_env(self):
        with patch.object(trainer, 'BombermanFFAEnv') as env, self.assertRaises(ValueError):
            trainer.train(policy_objective='invalid-objective')
        env.assert_not_called()

    def test_ratio_full_state_resume_matches_continuous_cpu_cuda(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            with tempfile.TemporaryDirectory(dir=ROOT/'runs', prefix='d0-resume-') as folder:
                base = Path(folder)
                def execute(name, pause=False, resume=False):
                    random.seed(11)
                    stop = threading.Event()
                    def record(row):
                        if pause and row['episode'] == 2:
                            stop.set()
                    with patch('builtins.print'):
                        trainer.train(episodes=4, board_size=9, max_steps=12, seed=23, device=device,
                            checkpoint_dir=str(base/name), resume_from=str(B), lr=1e-4,
                            policy_objective='ratio', save_training_state=True, validation_every=0,
                            run_report=False, stop_event=stop, on_episode=record,
                            resume_training_state=str(base/name/'training_state.pt') if resume else '')
                    return load_training_state(base/name/'training_state.pt')
                full = execute('full')
                partial = execute('paused', pause=True)
                self.assertEqual(partial['episode'], 2)
                continued = execute('paused', resume=True)
                def same(a, b, path='state'):
                    if isinstance(a, torch.Tensor):
                        self.assertTrue(torch.equal(a, b), path)
                    elif isinstance(a, dict):
                        self.assertEqual(a.keys(), b.keys())
                        for key in a: same(a[key], b[key], path + '.' + str(key))
                    elif isinstance(a, (list, tuple)):
                        self.assertEqual(len(a), len(b))
                        for i, (x, y) in enumerate(zip(a, b)): same(x, y, path + '.' + str(i))
                    else: self.assertEqual(a, b)
                same(full, continued)


class D0SmokeGuards(unittest.TestCase):
    def test_runner_refuses_overwrite_wrong_b_and_unavailable_device(self):
        import run_ppo_d0 as smoke
        with patch.object(Path, 'mkdir') as mkdir:
            with self.assertRaises(FileExistsError):
                smoke.run(ROOT)
            with patch.object(smoke, 'digest', return_value='wrong'):
                with self.assertRaises(ValueError):
                    smoke.run(ROOT/'unused-d0-smoke')
            with patch.object(torch.cuda, 'is_available', return_value=False):
                with self.assertRaises(ValueError):
                    smoke.run(ROOT/'unused-d0-smoke', 'cuda')
            mkdir.assert_not_called()


if __name__ == '__main__':
    unittest.main()
