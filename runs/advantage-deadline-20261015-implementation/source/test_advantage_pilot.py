"""T4-A1 numerical edges, default parity, loss isolation and paired protocol guards."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import torch
import torch.nn.functional as F

import run_advantage_pilot as runner
import run_combat_move_pilot as shared
import train_actor_critic as trainer
from test_combat_move_pilot import ready_arm, result


class AdvantageChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hand_calculation_population_std_and_detach_cpu_cuda(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            raw = torch.tensor([-2., 0., 2.], device=device, requires_grad=True)
            normalized, skipped = trainer.normalize_policy_advantages(raw)
            self.assertFalse(skipped)
            self.assertFalse(normalized.requires_grad)
            expected = torch.tensor([-1.2247448714, 0, 1.2247448714], device=device)
            self.assertTrue(torch.allclose(normalized, expected))
            self.assertTrue(torch.equal(raw.detach(), torch.tensor([-2., 0., 2.], device=device)))

    def test_short_constant_and_nearly_constant_keep_signal(self):
        for values in ([3.], [2., 2., 2.], [0., 1e-9]):
            raw = torch.tensor(values, requires_grad=True)
            normalized, skipped = trainer.normalize_policy_advantages(raw)
            self.assertTrue(skipped)
            self.assertFalse(normalized.requires_grad)
            self.assertTrue(torch.equal(raw.detach(), normalized))

    def test_long_vector_zero_mean_unit_std_and_finite(self):
        normalized, skipped = trainer.normalize_policy_advantages(torch.linspace(-100, 100, 500))
        self.assertFalse(skipped)
        self.assertLess(abs(normalized.mean().item()), 1e-6)
        self.assertAlmostEqual(normalized.std(unbiased=False).item(), 1, places=6)

    def test_nonfinite_advantage_fails_before_update(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(FloatingPointError):
                trainer.normalize_policy_advantages(torch.tensor([0., value]))

    def test_actor_normalization_does_not_change_critic_targets_or_gradients(self):
        targets = torch.tensor([1., 2., 4.])
        gradients = []
        for normalize in (False, True):
            values = torch.tensor([.2, .3, .4], requires_grad=True)
            log_probs = torch.tensor([-.5, -.3, -.7], requires_grad=True)
            advantage = targets - values.detach()
            if normalize:
                advantage, _ = trainer.normalize_policy_advantages(advantage)
            loss = -(log_probs * advantage).mean() + .5 * F.mse_loss(values, targets)
            loss.backward()
            gradients.append((values.grad.clone(), log_probs.grad.clone()))
        self.assertTrue(torch.equal(gradients[0][0], gradients[1][0]))
        self.assertFalse(torch.equal(gradients[0][1], gradients[1][1]))
        self.assertTrue(torch.equal(targets, torch.tensor([1., 2., 4.])))

    def test_default_and_diagnostics_keep_previous_weights_rewards_actions(self):
        before = runner.ROOT / 'runs/advantage-implementation-20261015/before/train_actor_critic.py'
        spec = importlib.util.spec_from_file_location('before_advantage_trainer', before)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        observed = []
        for fn, kwargs in ((old.train, {}), (trainer.train, {}),
                           (trainer.train, {'advantage_diagnostics': True})):
            rows, actions = [], []
            with patch('torch.save'), patch('os.makedirs'), patch('builtins.print'):
                net, rewards, wins = fn(episodes=2, board_size=9, num_agents=2, max_steps=12,
                    lr=1e-4, device='cpu', seed=runner.TRAIN_SEED, resume_from=str(runner.WARM_START),
                    validation_every=0, run_report=False, on_episode=rows.append,
                    on_step=lambda env, ep, aid, a, opp: actions.append((ep, aid, list(a))), **kwargs)
            observed.append((deepcopy(net.state_dict()), rewards, wins, rows, actions))
        for current in observed[1:]:
            self.assertEqual(observed[0][1:3], current[1:3])
            self.assertEqual(observed[0][4], current[4])
            self.assertTrue(all(torch.equal(v, current[0][k]) for k, v in observed[0][0].items()))
            self.assertEqual(observed[0][3], [{k: v for k, v in r.items() if k != 'advantage'} for r in current[3]])

    def test_one_update_changes_only_policy_loss_before_shared_gradient_cpu_cuda(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            observed = []
            for normalize in (False, True):
                rows, actions = [], []
                with patch('torch.save'), patch('os.makedirs'), patch('builtins.print'):
                    net, rewards, _ = trainer.train(episodes=1, board_size=9, max_steps=12,
                        lr=1e-4, device=device, seed=runner.TRAIN_SEED,
                        resume_from=str(runner.WARM_START), validation_every=0, run_report=False,
                        normalize_advantage=normalize, advantage_diagnostics=True, on_episode=rows.append,
                        on_step=lambda env, ep, aid, a, opp: actions.append(list(a)))
                observed.append((deepcopy(net.state_dict()), rewards, actions, rows[0]))
            self.assertEqual(observed[0][1:3], observed[1][1:3])
            for field in ('value_loss', 'escape_loss', 'combat_loss', 'navigation_loss', 'reward', 'move_guidance'):
                self.assertEqual(observed[0][3][field], observed[1][3][field])
            self.assertNotEqual(observed[0][3]['policy_loss'], observed[1][3]['policy_loss'])
            self.assertFalse(all(torch.equal(v, observed[1][0][k]) for k, v in observed[0][0].items()))
            diag = observed[1][3]['advantage']
            self.assertEqual(diag['samples'], 12)
            self.assertFalse(diag['normalization_skipped'])
            self.assertAlmostEqual(diag['policy']['mean'], 0, places=5)
            self.assertAlmostEqual(diag['policy']['std'], 1, places=5)
            self.assertEqual(sum(g['samples'] for g in diag['entropy_by_mask_size'].values()), 12)

    def test_single_step_training_retains_raw_policy_update(self):
        observed = []
        for normalize in (False, True):
            rows = []
            with patch('torch.save'), patch('os.makedirs'), patch('builtins.print'):
                net, _, _ = trainer.train(episodes=1, board_size=9, max_steps=1, device='cpu',
                    seed=runner.TRAIN_SEED, resume_from=str(runner.WARM_START), validation_every=0,
                    run_report=False, normalize_advantage=normalize, advantage_diagnostics=True,
                    on_episode=rows.append)
            observed.append((deepcopy(net.state_dict()), rows[0]))
        self.assertTrue(observed[1][1]['advantage']['normalization_skipped'])
        self.assertEqual(observed[0][1]['policy_loss'], observed[1][1]['policy_loss'])
        self.assertTrue(all(torch.equal(v, observed[1][0][k]) for k, v in observed[0][0].items()))


class ProtocolChecks(unittest.TestCase):
    def run_mocked(self, broken=None, smoke=False, edit=False):
        output = runner.ROOT / ('runs/uncreated-advantage-' + uuid4().hex)
        arms = {a: ready_arm(output, a) for a in runner.ARMS}
        if broken:
            arms['normalized'][broken] = False
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
            self.assertEqual(kwargs['test_seed'], 34000)
            return {n: result(100 if n == 'normalized' else 0, seed=34000)
                    for n in ('original', 'control', 'normalized')}
        with patch.object(runner, 'digest', side_effect=digest), patch.object(runner, 'fresh_test_check') as fresh, \
             patch.object(runner, 'BaseNet'), patch.object(runner.trainer, 'smart_load_weights'), \
             patch.object(runner, 'skill_probe', return_value={'records': []}), \
             patch.object(runner, 'train_arm', side_effect=train), \
             patch.object(runner, 'evaluate_models', side_effect=evaluate) as evaluation, \
             patch.object(runner, 'stage_timer', side_effect=lambda s: (threading.Event(), Mock())), \
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

    def test_paired_config_only_normalization_differs_and_selection_precedes_test(self):
        summary, order, saved, fresh, _ = self.run_mocked()
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['decision'], 'replicate_before_replacing')
        fresh.assert_called_once_with(34000)
        self.assertEqual(order[0], ('save', 'config.json'))
        configs = [entry[2] for entry in order if entry[0] == 'train']
        self.assertEqual(configs[0]['normalize_advantage'], False)
        self.assertEqual(configs[1]['normalize_advantage'], True)
        self.assertEqual({k: v for k, v in configs[0].items() if k != 'normalize_advantage'},
                         {k: v for k, v in configs[1].items() if k != 'normalize_advantage'})
        self.assertEqual(configs[0]['attack_teacher'], 'local')
        self.assertEqual(saved['config.json']['critic_target'], 'MC returns')

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

    def test_new_seed_range_and_cpu_evaluation_are_wired_without_teacher(self):
        with patch.object(Path, 'rglob', return_value=[Path('evaluation_seen.json')]), \
             patch.object(Path, 'read_text', return_value=json.dumps(result(seed=34000))):
            with self.assertRaisesRegex(ValueError, 'Test34000 already opened'):
                shared.fresh_test_check(runner.TEST_SEED)
        with patch.object(shared, 'make_ac_fn', return_value=object()) as load, \
             patch.object(shared, 'evaluate_seeded', side_effect=lambda *a, **k: result(seed=a[4])) as evaluate, \
             patch.object(shared, 'certify_combat_move') as teacher, \
             patch.object(shared, 'digest', return_value='test-only'), patch.object(shared, 'dump'):
            results = shared.evaluate_models(runner.ROOT / 'unused', threading.Event(), False,
                                            arm_names=runner.ARMS, test_seed=runner.TEST_SEED)
        self.assertEqual(set(results), {'original', 'control', 'normalized'})
        self.assertTrue(all(c.args[3:6] == (400, 34000, 500) for c in evaluate.call_args_list))
        self.assertTrue(all(c.args[1:] == (9, 'cpu') for c in load.call_args_list))
        teacher.assert_not_called()


if __name__ == '__main__':
    unittest.main()
