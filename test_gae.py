"""T4-B0: GAE math, boundaries, frozen-update parity and MC compatibility."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import train_actor_critic as trainer
from test_combat import arena, put_bomb
from benchmark_t3 import observations
from legacy_test_support import legacy_train, strip_sd

ROOT = Path(__file__).resolve().parent
B = ROOT / 'runs/pressure-C-next/final_phaseC.pt'


def inputs(rewards=(1., 2., 3.), values=(.5, 1., 1.5), next_values=(1., 1.5, 999.),
           bootstrap=(True, True, False), continuation=(True, True, False), device='cpu', dtype=torch.float64):
    return [torch.tensor(v, dtype=dtype, device=device) for v in (rewards, values, next_values)] + [
        torch.tensor(v, dtype=torch.bool, device=device) for v in (bootstrap, continuation)]


def mc_returns(rewards, gamma):
    result, carry = [], 0.
    for reward in reversed(rewards):
        carry = float(reward) + gamma * carry
        result.insert(0, carry)
    return result


class GAEMathChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hand_calculation_and_lambda_zero_cpu_cuda(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            tensors = inputs(device=device)
            advantage, delta = trainer.gae_advantages(*tensors, gamma=.9, gae_lambda=.5)
            torch.testing.assert_close(delta, torch.tensor([1.4, 2.35, 1.5], dtype=torch.float64, device=device))
            torch.testing.assert_close(advantage, torch.tensor([2.76125, 3.025, 1.5], dtype=torch.float64, device=device))
            one_step, same_delta = trainer.gae_advantages(*tensors, gamma=.9, gae_lambda=0)
            self.assertTrue(torch.equal(one_step, same_delta))

    def test_lambda_one_matches_mc_one_step_and_500_step_trajectories(self):
        rng = np.random.default_rng(2026101503)
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            for length in (1, 3, 500):
                rewards = rng.normal(size=length)
                rewards[-1] += 14  # delayed owned kill / win reward
                values = rng.normal(size=length)
                args = inputs(rewards, values, [*values[1:], 99999],
                    [True] * (length - 1) + [False], [True] * (length - 1) + [False], device=device)
                before = [t.clone() for t in args]
                advantage, _ = trainer.gae_advantages(*args, gamma=.99, gae_lambda=1)
                expected = torch.tensor(mc_returns(rewards, .99) - values, device=device, dtype=torch.float64)
                torch.testing.assert_close(advantage, expected, rtol=1e-10, atol=1e-10)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(args, before)))

    def test_artificial_cut_uses_correct_next_value_without_next_advantage(self):
        args = inputs([1., 2., 100., 200.], [.1, .2, 10., 20.], [.2, 42., 20., 999.],
                      [True, True, True, False], [True, False, True, False])
        advantage, _ = trainer.gae_advantages(*args, gamma=.9, gae_lambda=1)
        torch.testing.assert_close(advantage, torch.tensor([36.72, 39.6, 270., 180.], dtype=torch.float64))
        args[0][2:] *= 1000  # Changing the later reset/segment must not affect the prefix.
        changed, _ = trainer.gae_advantages(*args, gamma=.9, gae_lambda=1)
        self.assertTrue(torch.equal(advantage[:2], changed[:2]))

    def test_terminal_does_not_bootstrap_or_leak_into_reset(self):
        args = inputs([1., 2., 100., 200.], [.1, .2, 10., 20.], [.2, 42., 20., 999.],
                      [True, False, True, False], [True, False, True, False])
        advantage, _ = trainer.gae_advantages(*args, gamma=.9, gae_lambda=1)
        torch.testing.assert_close(advantage, torch.tensor([2.7, 1.8, 270., 180.], dtype=torch.float64))

    def test_zero_discount_and_single_terminal_death(self):
        args = inputs([-2.5], [.7], [999.], [False], [False])
        for gamma in (0, .99, 1):
            advantage, _ = trainer.gae_advantages(*args, gamma=gamma, gae_lambda=.95)
            torch.testing.assert_close(advantage, torch.tensor([-3.2], dtype=torch.float64))

    def test_advantage_and_targets_detach_but_critic_receives_gradient(self):
        value_gradients = []
        for estimator in ('mc', 'gae'):
            args = inputs()
            for t in args[:3]:
                t.requires_grad_()
            targets = torch.tensor(mc_returns(args[0].detach(), .9), dtype=torch.float64)
            if estimator == 'gae':
                advantage, delta = trainer.gae_advantages(*args, gamma=.9)
                self.assertFalse(delta.requires_grad)
            else:
                advantage = targets - args[1].detach()
            self.assertFalse(advantage.requires_grad)
            log_probs = torch.tensor([-.4, -.8, -.5], dtype=torch.float64, requires_grad=True)
            loss = -(log_probs * advantage).mean() + .5 * F.mse_loss(args[1], targets)
            loss.backward()
            self.assertIsNone(args[0].grad)
            self.assertIsNone(args[2].grad)
            self.assertIsNotNone(log_probs.grad)
            value_gradients.append(args[1].grad)
        self.assertTrue(torch.equal(*value_gradients))

    def test_invalid_parameters_shapes_masks_and_nonfinite_inputs_fail(self):
        for params in ({'gamma': -1}, {'gamma': float('nan')}, {'gae_lambda': 1.1}, {'gae_lambda': float('inf')}):
            with self.assertRaises(ValueError):
                trainer.gae_advantages(*inputs(), **params)
        for index, value in ((0, torch.tensor([])), (1, torch.zeros(2, dtype=torch.float64)),
                             (2, torch.zeros(3, dtype=torch.float32)),
                             (3, torch.ones(3)), (4, torch.tensor([True, True, True])),
                             (3, torch.tensor([False, True, False]))):
            args = inputs()
            args[index] = value
            with self.assertRaises(ValueError):
                trainer.gae_advantages(*args)

        args = inputs()
        args[0] = [1, 2, 3]
        with self.assertRaises(TypeError):
            trainer.gae_advantages(*args)
        for index in range(3):
            args = inputs()
            args[index][0] = float('nan')
            with self.assertRaises(FloatingPointError):
                trainer.gae_advantages(*args)
        if torch.cuda.is_available():
            args = inputs()
            args[2] = args[2].cuda()
            with self.assertRaises(ValueError):
                trainer.gae_advantages(*args)

    def test_helper_preserves_numpy_torch_rng_and_input_state(self):
        args = inputs()
        numpy_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
        cuda_state = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
        before = [t.clone() for t in args]
        trainer.gae_advantages(*args)
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        if cuda_state is not None:
            self.assertTrue(torch.equal(cuda_state, torch.cuda.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(args, before)))


class GAETrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def observe(self, *, mode='mc', lam=1., device='cpu', env=None, max_steps=20, stop=None, hook=None):
        rows, actions, gradients, calls = [], [], [], []
        real_clip, real_gae = torch.nn.utils.clip_grad_norm_, trainer.gae_advantages
        def clip(parameters, *args, **kwargs):
            parameters = list(parameters)
            gradients.append([p.grad.detach().clone() if p.grad is not None else None for p in parameters])
            return real_clip(parameters, *args, **kwargs)
        def gae(*args, **kwargs):
            result = real_gae(*args, **kwargs)
            calls.append(([t.detach().cpu().clone() for t in args[:5]], result[0].cpu().clone()))
            return result
        def frame(current, ep, aid, sampled, opponents):
            if current.step_count:
                actions.append((aid, list(sampled)))
            if hook is not None:
                hook(current, ep, aid, sampled, opponents)
        with patch('torch.save'), patch('os.makedirs'), patch('builtins.print'), \
             patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=clip), \
             patch.object(trainer, 'gae_advantages', side_effect=gae):
            kwargs = dict(episodes=1, board_size=9 if env is None else env.board_size,
                num_agents=2 if env is None else env.num_agents, max_steps=max_steps, lr=1e-4,
                device=device, seed=2026101503, resume_from=str(B) if env is None else '',
                checkpoint_dir='unused-gae-unit', run_report=False, validation_every=0,
                advantage_estimator=mode, gae_lambda=lam, advantage_diagnostics=True,
                stop_event=stop, on_step=frame, on_episode=rows.append)
            if env is None:
                net, rewards, wins = trainer.train(**kwargs)
            else:
                with patch.object(trainer, 'BombermanFFAEnv', return_value=env), \
                     patch.object(env, 'reset', side_effect=lambda: observations(env)), \
                     patch.object(trainer.RuleBasedOpponent, 'select_action_env', return_value=0), \
                     patch('torch.distributions.Categorical.sample', side_effect=lambda: torch.tensor([0], device=device)):
                    net, rewards, wins = trainer.train(**kwargs)
        return {'state': deepcopy(net.state_dict()), 'rows': rows, 'actions': actions,
                'gradients': gradients, 'calls': calls, 'rewards': rewards, 'wins': wins}

    def test_frozen_first_update_lambda_one_matches_mc_losses_gradients_adam_cpu_cuda(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            mc, gae = self.observe(device=device), self.observe(mode='gae', device=device)
            self.assertEqual(mc['actions'], gae['actions'])
            self.assertEqual(mc['rewards'], gae['rewards'])
            for field in ('value_loss', 'escape_loss', 'combat_loss', 'navigation_loss', 'reward', 'move_guidance'):
                self.assertEqual(mc['rows'][0][field], gae['rows'][0][field])
            for field in ('policy_loss', 'loss'):
                self.assertAlmostEqual(mc['rows'][0][field], gae['rows'][0][field], delta=1e-5)
            for a, b in zip(mc['gradients'][0], gae['gradients'][0]):
                if a is None:
                    self.assertIsNone(b)
                else:
                    torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6)
            for key, value in mc['state'].items():
                # Adam amplifies float32 rounding near zero gradients; default MC is tested bit-exactly.
                torch.testing.assert_close(value, gae['state'][key], rtol=1e-4, atol=5e-6)
            args, advantage = gae['calls'][0]
            target = torch.tensor(mc_returns(args[0], .99), dtype=args[1].dtype) - args[1]
            torch.testing.assert_close(advantage, target, rtol=1e-5, atol=1e-5)
            self.assertEqual(gae['rows'][0]['advantage']['last_boundary'], 'time_limit')

    def test_lambda_point95_keeps_critic_and_auxiliary_targets_on_same_data(self):
        mc, gae = self.observe(), self.observe(mode='gae', lam=.95)
        self.assertEqual(mc['actions'], gae['actions'])
        for field in ('value_loss', 'escape_loss', 'combat_loss', 'navigation_loss', 'reward'):
            self.assertEqual(mc['rows'][0][field], gae['rows'][0][field])
        self.assertNotEqual(mc['rows'][0]['policy_loss'], gae['rows'][0]['policy_loss'])
        self.assertEqual(mc['rows'][0]['advantage']['explained_variance'], gae['rows'][0]['advantage']['explained_variance'])

    def test_default_mc_matches_before_source_actions_rewards_weights_and_all_metrics(self):
        path = ROOT / 'runs/gae-b0-implementation-20261015/before/train_actor_critic.py'
        spec = importlib.util.spec_from_file_location('before_gae_trainer', path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        outputs = []
        for fn in (old.train, legacy_train):
            rows, actions = [], []
            with patch('torch.save'), patch('os.makedirs'), patch('builtins.print'):
                net, rewards, wins = fn(episodes=2, board_size=9, max_steps=20, lr=1e-4, device='cpu',
                    seed=2026101503, resume_from=str(B), validation_every=0, run_report=False,
                    advantage_diagnostics=True, on_episode=rows.append,
                    on_step=lambda env, ep, aid, a, opp: actions.append((ep, aid, list(a))))
            outputs.append((deepcopy(net.state_dict()), rewards, wins, strip_sd(rows), actions))
        self.assertEqual(outputs[0][1:], outputs[1][1:])
        self.assertTrue(all(torch.equal(v, outputs[1][0][k]) for k, v in outputs[0][0].items()))

    def test_delayed_owned_kill_and_win_end_the_learner_segment(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 1, 5
        put_bomb(env, 1, 3, 0, timer=3, blast_range=2)
        result = self.observe(mode='gae', env=env, max_steps=10)
        row, (args, advantage) = result['rows'][0], result['calls'][0]
        self.assertEqual(row['active_win'], 1)
        self.assertEqual(row['learner_steps'], 3)
        self.assertEqual(row['advantage']['last_boundary'], 'match_end')
        self.assertGreater(args[0][-1].item(), 10)
        self.assertFalse(args[3][-1])
        self.assertFalse(args[4][-1])
        torch.testing.assert_close(advantage, torch.tensor(mc_returns(args[0], .99)) - args[1], rtol=1e-5, atol=1e-5)

    def test_learner_death_ends_buffer_while_ffa_match_continues(self):
        env = arena(3)
        env.max_steps = 4
        put_bomb(env, 1, 1, 1, timer=1, blast_range=1)
        result = self.observe(mode='gae', env=env, max_steps=4)
        row, (args, advantage) = result['rows'][0], result['calls'][0]
        self.assertEqual(env.step_count, 4)
        self.assertEqual(sum(a.alive for a in env.agents), 2)
        self.assertEqual(row['learner_steps'], 1)
        self.assertEqual(row['advantage']['last_boundary'], 'learner_death')
        self.assertEqual(args[0].tolist(), [-1.])
        self.assertFalse(args[3].any())
        self.assertFalse(args[4].any())
        torch.testing.assert_close(advantage, args[0] - args[1])

    def test_timeout_500_with_both_alive_does_not_bootstrap(self):
        env = arena(2)
        env.max_steps = 500
        result = self.observe(mode='gae', env=env, max_steps=500)
        row, (args, _) = result['rows'][0], result['calls'][0]
        self.assertEqual(row['learner_steps'], 500)
        self.assertTrue(all(a.alive for a in env.agents))
        self.assertEqual(row['advantage']['last_boundary'], 'time_limit')
        self.assertEqual(args[2][-1], 0)
        self.assertFalse(args[3][-1])
        self.assertFalse(args[4][-1])

    def test_cancelled_partial_episode_makes_no_update(self):
        stop = threading.Event()
        result = self.observe(mode='gae', max_steps=20, stop=stop,
            hook=lambda env, *a: stop.set() if env.step_count == 2 else None)
        self.assertFalse(result['rows'])
        self.assertFalse(result['gradients'])
        self.assertFalse(result['calls'])
        self.assertFalse(result['rewards'])

    def test_bad_estimator_and_lambda_fail_before_env_or_weights(self):
        with patch.object(trainer, 'BombermanFFAEnv') as env:
            for params in ({'advantage_estimator': 'ppo'}, {'gae_lambda': float('nan')},
                           {'advantage_estimator': 'gae', 'gamma': -1}):
                with self.assertRaises(ValueError):
                    trainer.train(**params)
            env.assert_not_called()


class GAESmokeGuards(unittest.TestCase):
    def test_smoke_refuses_overwrite_wrong_b_and_unavailable_device_before_writes(self):
        import run_gae_b0 as smoke
        with patch.object(Path, 'mkdir') as mkdir:
            with self.assertRaises(FileExistsError):
                smoke.run(ROOT)
            with patch.object(smoke, 'digest', return_value='wrong-B'):
                with self.assertRaises(ValueError):
                    smoke.run(ROOT / 'unused-gae-smoke')
            with patch.object(torch.cuda, 'is_available', return_value=False):
                with self.assertRaises(ValueError):
                    smoke.run(ROOT / 'unused-gae-smoke', 'cuda')
            mkdir.assert_not_called()


if __name__ == '__main__':
    unittest.main()
