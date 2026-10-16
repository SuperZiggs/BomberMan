"""T4-R1 reward isolation, unchanged inference and paired protocol guards."""
from copy import deepcopy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import pickle
import threading
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import torch
import numpy as np
import torch.nn.functional as F

import run_reward_pilot as runner
import run_combat_move_pilot as shared
import train_actor_critic as trainer
import evaluate as evaluator
from benchmark_t3 import observations
from bomberman_env import WOODEN_WALL
from test_combat import arena, put_bomb
from test_combat_move_pilot import ready_arm, result
from legacy_test_support import legacy_train, strip_sd


class ProtocolChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @unittest.skip('closed R0/R1 pilot pins the pre-merge bomberman_env.py hash; reproduce at commit 860753f')
    def test_preflight_reprices_all_original_r0_steps_and_honors_cancellation(self):
        stop = threading.Event()
        report = runner.preflight(stop)
        self.assertTrue(report['passed'])
        self.assertEqual(report['transitions'], 39338)
        self.assertGreater(report['eligible_walls'], 0)
        self.assertLessEqual(report['max_delta_error'], 1e-10)
        stop.set()
        self.assertIsNone(runner.preflight(stop))

    def run_mocked(self, broken=None, smoke=False, edit=False, phase_calls=None, clock=None):
        output = runner.ROOT / ('runs/uncreated-reward-' + uuid4().hex)
        arms = {a: ready_arm(output, a) for a in runner.ARMS}
        if broken:
            arms['half_wall'][broken] = False
        changed = False
        saved, order = {}, []
        def train(*args, **kwargs):
            nonlocal changed
            order.append(('train', args[1], kwargs))
            changed = edit
            return arms[args[1]]
        def save(path, data):
            saved[path.name] = data
            order.append(('save', path.name))
        def digest(path):
            if path == runner.WARM_START:
                return runner.B_SHA256
            return 'changed' if changed and path.name == 'train_actor_critic.py' else 'test-only-source'
        def evaluate(*args, **kwargs):
            self.assertIn('selected_models.json', saved)
            self.assertIn('test_opened.json', saved)
            self.assertEqual(kwargs['test_seed'], 36000)
            return {n: result(100 if n == 'half_wall' else 0, seed=36000)
                    for n in ('original', 'control', 'half_wall')}
        def timer(seconds):
            if phase_calls is not None:
                phase_calls.append(seconds)
            return threading.Event(), Mock()
        with patch.object(runner, 'digest', side_effect=digest), patch.object(runner, 'fresh_test_check') as fresh, \
             patch.object(runner, 'run_lock', return_value=nullcontext()), \
             patch.object(runner, 'preflight', return_value={'passed': True}), \
             patch.object(runner, 'BaseNet'), patch.object(runner.trainer, 'smart_load_weights'), \
             patch.object(runner, 'skill_probe', return_value={'records': []}), \
             patch.object(runner, 'train_arm', side_effect=train), \
             patch.object(runner, 'evaluate_models', side_effect=evaluate) as evaluation, \
             patch.object(runner, 'stage_timer', side_effect=timer), \
             patch.object(runner, 'perf_counter', side_effect=clock or runner.perf_counter), \
             patch.object(Path, 'mkdir'), patch.object(Path, 'write_text'), \
             patch.object(runner.shutil, 'copy2'), patch.object(runner.shutil, 'make_archive'), \
             patch.object(runner, 'dump', side_effect=save), patch('builtins.print'):
            if edit:
                with self.assertRaises(AssertionError):
                    runner.run(output)
                summary = saved['summary.json']
            else:
                summary = runner.run(output, smoke=smoke)
        return summary, order, saved, fresh, evaluation

    def test_paired_config_only_wall_differs_and_selection_precedes_test(self):
        summary, order, saved, fresh, _ = self.run_mocked()
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['decision'], 'replicate_before_replacing')
        fresh.assert_called_once_with(36000)
        self.assertEqual(order[0], ('save', 'config.json'))
        configs = [entry[2] for entry in order if entry[0] == 'train']
        self.assertEqual([c['wall_destroy_bonus'] for c in configs], [.15, .075])
        self.assertEqual({k: v for k, v in configs[0].items() if k != 'wall_destroy_bonus'},
                         {k: v for k, v in configs[1].items() if k != 'wall_destroy_bonus'})
        for c in configs:
            self.assertFalse(c['normalize_advantage'])
            self.assertEqual(c['advantage_estimator'], 'mc')
            self.assertEqual(c['attack_teacher'], 'local')
            self.assertTrue(c['reward_diagnostics'] and c['contact_diagnostics'])
        self.assertEqual(saved['config.json']['critic_target'], 'MC returns')
        self.assertTrue(saved['preflight.json']['passed'])

    def test_incomplete_arms_and_changed_source_never_open_test(self):
        for broken in ('training_complete', 'validation_complete', 'skill_probes_complete'):
            summary, _, _, _, evaluation = self.run_mocked(broken=broken)
            self.assertEqual(summary['status'], 'incomplete')
            evaluation.assert_not_called()
        summary, _, _, _, evaluation = self.run_mocked(edit=True)
        self.assertFalse(summary['hashes_unchanged'])
        self.assertEqual(summary['decision'], 'retain_original')
        evaluation.assert_not_called()

    def test_smoke_never_eligible(self):
        summary, _, _, fresh, _ = self.run_mocked(smoke=True)
        self.assertFalse(summary['full_protocol'])
        self.assertFalse(summary['eligible_for_replication'])
        fresh.assert_not_called()

    def test_budget_clobber_and_gpu_validation(self):
        with self.assertRaises(ValueError):
            runner.run(runner.ROOT / 'unused', budget_minutes=151)
        with self.assertRaises(FileExistsError):
            runner.run(runner.ROOT)
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaises(ValueError):
                runner.run(runner.ROOT / 'unused', device='cuda')

    def test_training_cap_100_and_evaluation_clipped_to_overall_150(self):
        phases = []
        summary, _, saved, _, _ = self.run_mocked(phase_calls=phases,
            clock=[0, 0, 12 * 60, 112 * 60, 130 * 60])
        self.assertEqual(phases, [15 * 60, 100 * 60, 38 * 60])
        self.assertEqual(saved['config.json']['budget_minutes']['training'], 100)
        self.assertEqual(summary['elapsed_seconds'], 130 * 60)

    def test_new_seed_range_and_cpu_evaluation_are_wired_without_teacher(self):
        with patch.object(Path, 'rglob', return_value=[Path('evaluation_seen.json')]), \
             patch.object(Path, 'read_text', return_value=json.dumps(result(seed=36000))):
            with self.assertRaisesRegex(ValueError, 'Test36000 already opened'):
                shared.fresh_test_check(runner.TEST_SEED)
        with patch.object(shared, 'make_ac_fn', return_value=object()) as load, \
             patch.object(shared, 'evaluate_seeded', side_effect=lambda *a, **k: result(seed=a[4])) as evaluate, \
             patch.object(shared, 'certify_combat_move') as teacher, \
             patch.object(shared, 'digest', return_value='test-only'), patch.object(shared, 'dump'):
            results = shared.evaluate_models(runner.ROOT / 'unused', threading.Event(), False,
                                            arm_names=runner.ARMS, test_seed=runner.TEST_SEED,
                                            contact_diagnostics=True)
        self.assertEqual(set(results), {'original', 'control', 'half_wall'})
        self.assertTrue(all(c.args[3:6] == (400, 36000, 500) for c in evaluate.call_args_list))
        self.assertTrue(all(c.args[1:] == (9, 'cpu') for c in load.call_args_list))
        self.assertTrue(all(c.kwargs['contact_diagnostics'] for c in evaluate.call_args_list))
        teacher.assert_not_called()

    def test_interrupted_test_range_is_reserved_even_without_evaluation_file(self):
        marker = {'seed': 36000, 'num_episodes': 400, 'num_agents': 2}
        with patch.object(Path, 'rglob', side_effect=[[], [Path('test_opened.json')]]), \
             patch.object(Path, 'read_text', return_value=json.dumps(marker)):
            with self.assertRaisesRegex(ValueError, 'already opened'):
                shared.fresh_test_check(36000)


class RewardIsolationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_actions_change_only_eligible_wall_reward_and_preserve_physics(self):
        for dead in (False, True):
            outputs = []
            for coefficient in (.15, .075):
                env = arena(2)
                env.max_steps = 2
                env.agents[0].row, env.agents[0].col = (3, 3) if dead else (3, 1)
                env.grid[3, 4] = WOODEN_WALL
                put_bomb(env, 3, 3, 0, timer=1, blast_range=1)
                rows, actions = [], []
                with patch.object(trainer, 'BombermanFFAEnv', return_value=env), \
                     patch.object(env, 'reset', side_effect=lambda: observations(env)), \
                     patch.object(trainer.RuleBasedOpponent, 'select_action_env', return_value=0), \
                     patch('torch.distributions.Categorical.sample', side_effect=lambda: torch.tensor([0])), \
                     patch('torch.save'), patch('os.makedirs'), patch('builtins.print'):
                    net, _, _ = trainer.train(episodes=1, board_size=7, num_agents=2, max_steps=2,
                        device='cpu', lr=0, seed=7, validation_every=0, run_report=False,
                        wall_destroy_bonus=coefficient, reward_diagnostics=True, on_episode=rows.append,
                        on_step=lambda e, ep, aid, a, opp: actions.append((e.step_count, list(a))))
                outputs.append((rows[0], actions, pickle.dumps(env.__dict__), deepcopy(net.state_dict())))
            a, b = outputs
            self.assertEqual(a[1:3], b[1:3])
            self.assertTrue(all(torch.equal(v, b[3][key]) for key, v in a[3].items()))
            self.assertAlmostEqual(b[0]['reward'] - a[0]['reward'], 0 if dead else -.075, places=12)
            self.assertEqual({k: v for k, v in a[0]['reward_components'].items() if k != 'wall'},
                             {k: v for k, v in b[0]['reward_components'].items() if k != 'wall'})

    def test_contact_observer_pure_conditional_and_resettable(self):
        env, state = arena(2), {}
        before = pickle.dumps(env.__dict__)
        np_before, torch_before = pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
        evaluator.observe_contact(env, 0, state)
        self.assertEqual(before, pickle.dumps(env.__dict__))
        self.assertEqual(np_before, pickle.dumps(np.random.get_state()))
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertIsNone(state['first_close_step'])
        self.assertFalse(state['left_spawn'])
        env.agents[0].row, env.agents[0].col, env.step_count = 3, 4, 2
        env.agents[1].row, env.agents[1].col = 3, 5
        evaluator.observe_contact(env, 0, state)
        self.assertEqual(state['first_close_step'], 2)
        self.assertTrue(state['left_spawn'])
        missing = {'contact': {'first_close_step': None, 'left_spawn': False}}
        self.assertIsNone(evaluator.summarize_contact([missing])['avg_first_close_step_conditional'])
        stats = evaluator.summarize_contact([missing, {'contact': state}])
        self.assertEqual(stats['reached_close_rate'], 50)
        self.assertEqual(stats['avg_first_close_step_conditional'], 2)

    def test_logging_matches_legacy_evaluator_including_sampled_actions_and_rng(self):
        spec = importlib.util.spec_from_file_location('before_r1_evaluator',
            runner.ROOT / 'runs/r1-implementation-20261015/before/evaluate.py')
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        policy = evaluator.make_ac_fn(str(runner.WARM_START), 9, 'cpu')
        outcomes = []
        for evaluate, diagnostics in ((old.evaluate_seeded, False), (evaluator.evaluate_seeded, False),
                                      (evaluator.evaluate_seeded, True)):
            actions = []
            def observed(env, obs, aid):
                action = policy(env, obs, aid)
                actions.append((env.step_count, aid, action))
                return action
            before_np, before_torch = pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
            kwargs = {'contact_diagnostics': True} if diagnostics else {}
            result = evaluate(observed, 9, 2, 4, 21000, 50, **kwargs)
            self.assertEqual(before_np, pickle.dumps(np.random.get_state()))
            self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
            if diagnostics:
                self.assertTrue(all('contact' in game and 'walls_rewarded' in game for game in result['games']))
                for record in [result, *result['by_corner'].values(), *result['games']]:
                    for key in ('contact', 'walls', 'walls_rewarded', 'avg_walls', 'avg_walls_rewarded'):
                        record.pop(key, None)
            outcomes.append((strip_sd(result), actions))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], outcomes[2])


if __name__ == '__main__':
    unittest.main()
