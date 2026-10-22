"""Actual bounded training, independent KL oracle and exact continuation."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from test_s2_kl_support import (ROOT, S2_REL, S2_SHA, assert_same, function, metrics,
    model_descriptor, read_json, read_state, register_case, rng_snapshot, sha)


class ScheduleTests(unittest.TestCase):
    def test_two_arms_share_every_paired_board_corner_and_only_rulebot(self):
        schedule = function(self, 'build_train_schedule')
        for smoke, count, first in ((False, 500, 710000), (True, 4, 790000)):
            left = schedule('control', smoke=smoke); right = schedule('anchor', smoke=smoke)
            self.assertIsInstance(left, list); self.assertEqual(len(left), count)
            self.assertEqual([(r['episode'], r['board_seed'], r['learner_corner']) for r in left],
                             [(r['episode'], r['board_seed'], r['learner_corner']) for r in right])
            for index, row in enumerate(left):
                self.assertEqual((row['episode'], row['board_seed'], row['learner_corner']),
                                 (index + 1, first + index // 2, index % 2))
                self.assertEqual(row['opponent_id'], 'RB')
            self.assertTrue(all(row['opponent_id'] == 'RB' for row in right))
        with self.assertRaises(ValueError): schedule('pool')


class KLMathTests(unittest.TestCase):
    """Regression PASS is honest: these exercise existing trainer mathematics."""
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_actual_train_kl_is_alive_step_mean_learner_to_masked_frozen_reference(self):
        import train_actor_critic as trainer
        original_copy = trainer.deepcopy
        originals, references, learner_logits, reference_logits, masks, rows, checks = [], [], [], [], [], [], []
        original_mask = trainer.get_action_mask
        def copied(value, *args, **kwargs):
            result = original_copy(value, *args, **kwargs)
            if isinstance(value, trainer.ActorCriticNet):
                originals.append(value); references.append(result)
                value.register_forward_hook(lambda module, inputs, output: learner_logits.append(output[0].detach().clone()))
                def reference_hook(module, inputs, output):
                    reference_logits.append(output[0].detach().clone())
                    checks.append((module.training, any(p.requires_grad for p in module.parameters()), torch.is_grad_enabled()))
                result.register_forward_hook(reference_hook)
            return result
        def mask(env, aid):
            value = original_mask(env, aid); masks.append(torch.tensor(value, dtype=torch.bool))
            return value
        cursor = 0
        def episode(row):
            nonlocal cursor
            count = row['advantage']['samples']
            local = []
            for learner, reference, legal in zip(learner_logits[cursor:cursor+count],
                                                 reference_logits[cursor:cursor+count], masks[cursor:cursor+count]):
                lp = F.log_softmax(learner.masked_fill(~legal, -1e9), -1)
                lq = F.log_softmax(reference.masked_fill(~legal, -1e9), -1)
                local.append((lp.exp() * (lp-lq)).sum())
            self.assertEqual(len(local), count)
            self.assertAlmostEqual(row['anchor_kl'], torch.stack(local).mean().item(), places=7)
            rows.append(row); cursor += count
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-math-') as tmp, patch.object(
                trainer, 'deepcopy', copied), patch.object(trainer, 'get_action_mask', mask):
            net, _, _ = trainer.train(phase='C', episodes=3, board_size=9, num_agents=2, max_steps=20,
                lr=.001, device='cpu', seed=2026102301, resume_from=str(ROOT / S2_REL),
                checkpoint_dir=tmp, validation_every=0, run_report=False, anchor_coef=1.,
                advantage_diagnostics=True, on_episode=episode)
        self.assertEqual(len(references), 1)
        self.assertTrue(checks); self.assertTrue(all(check == (False, False, False) for check in checks))
        self.assertTrue(all(p.grad is None for p in references[0].parameters()))
        s2 = torch.load(ROOT / S2_REL, map_location='cpu', weights_only=True)
        assert_same(self, references[0].state_dict(), s2, 'frozen S2')
        self.assertTrue(any(not torch.equal(value, s2[key]) for key, value in net.state_dict().items()))
        self.assertTrue(any(not bool(mask.all()) for mask in masks), 'Mask case must exclude an action')
        self.assertGreater(rows[-1]['anchor_kl'], 0.)
        forward, reverse = [], []
        for learner, reference, legal in zip(learner_logits, reference_logits, masks):
            lp = F.log_softmax(learner.masked_fill(~legal, -1e9), -1)
            lq = F.log_softmax(reference.masked_fill(~legal, -1e9), -1)
            forward.append((lp.exp() * (lp-lq)).sum().item())
            reverse.append((lq.exp() * (lq-lp)).sum().item())
        self.assertGreater(max(abs(a-b) for a,b in zip(forward, reverse)), 1e-8,
                           'Numerical fixture must distinguish KL directions')

    def test_beta_zero_disables_reference_and_diagnostics_are_rng_loss_weight_neutral(self):
        import train_actor_critic as trainer
        results = []
        for diagnostics in (False, True):
            rows, actions = [], []
            with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-observer-') as tmp:
                net, rewards, wins = trainer.train(phase='C', episodes=2, board_size=9, num_agents=2,
                    max_steps=12, device='cpu', seed=2026102301, resume_from=str(ROOT / S2_REL),
                    checkpoint_dir=tmp, validation_every=0, run_report=False, anchor_coef=0.,
                    advantage_diagnostics=diagnostics, on_episode=rows.append,
                    on_step=lambda env, ep, aid, acts, opp: actions.append((ep, aid, list(acts))))
                results.append((deepcopy(net.state_dict()), rewards, wins, actions, rows, rng_snapshot()))
            self.assertTrue(all('anchor_kl' not in row for row in rows))
        for key in range(4): assert_same(self, results[0][key], results[1][key], str(key))
        assert_same(self, [r['loss'] for r in results[0][4]], [r['loss'] for r in results[1][4]])
        assert_same(self, results[0][5], results[1][5], 'observer RNG')


class ExactResumeTests(unittest.TestCase):
    device = 'cpu'
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def continuation(self, arm, when, *, observe=False):
        train = function(self, 'train_arm')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-resume-') as tmp:
            _, full, _ = register_case(self, Path(tmp) / 'full', device=self.device)
            inputs, split, reg = register_case(self, Path(tmp) / 'split', device=self.device)
            self.assertEqual(train(full, arm), dict(arm=arm, episode=4, target=4, complete=True))
            class Interrupted(RuntimeError): pass
            def hook(stage, episode):
                if stage == when and episode == 2: raise Interrupted('acceptance interrupted commit')
            with self.assertRaises(Interrupted): train(split, arm, commit_hook=hook)
            partial = read_state(split, arm)
            self.assertEqual(partial['episode'], 1 if when == 'before_state_commit' else 2)
            for field in ('model', 'optimizer', 'scheduler', 'rng', 'histories', 'context', 'provenance'):
                self.assertIn(field, partial)
            self.assertIn(S2_SHA, str(partial['provenance']))
            self.assertIn(reg['registration_digest'], str(partial['provenance']))
            if observe:
                diagnose = function(self, 'diagnose_model')
                before_rng = rng_snapshot(); before_checkpoint = sha(split / 'arms' / arm / 'training_state.pt')
                diagnose(model_descriptor(checkpoint=inputs / S2_REL), reg)
                assert_same(self, before_rng, rng_snapshot(), 'offline diagnostics RNG')
                self.assertEqual(sha(split / 'arms' / arm / 'training_state.pt'), before_checkpoint)
            self.assertEqual(train(split, arm, resume=True), dict(arm=arm, episode=4, target=4, complete=True))
            a, b = read_state(full, arm), read_state(split, arm)
            for field in ('episode', 'model', 'optimizer', 'scheduler', 'rng', 'histories'):
                assert_same(self, a[field], b[field], field)
            for optional in ('curriculum_rng', 'group_rng'):
                self.assertEqual(optional in a, optional in b)
                if optional in a: assert_same(self, a[optional], b[optional], optional)
            for output in (full, split):
                rows = metrics(output, arm)
                self.assertEqual([r['episode'] for r in rows], [1, 2, 3, 4])
                self.assertEqual(len(read_state(output, arm)['histories']['reward']), 4)
                self.assertEqual(read_state(output, arm)['scheduler']['T_max'], 4)
                self.assertAlmostEqual(read_state(output, arm)['scheduler']['eta_min'], 2.5e-5*.05)
                for row in rows:
                    for key in ('policy_loss','value_loss','escape_loss','combat_loss','navigation_loss','ring_safety_loss','advantage'):
                        self.assertIn(key,row)
                    self.assertIn('grad_norm_before_clip', row['advantage'])
                    if arm == 'anchor': self.assertIn('anchor_kl', row)
                    else: self.assertNotIn('anchor_kl', row)
                self.assertEqual(sha(output / 'arms' / arm / 'checkpoints/0.pt'), S2_SHA)

    def test_cpu_both_arms_replay_uncommitted_episode_exactly(self):
        for arm in ('control', 'anchor'):
            with self.subTest(arm=arm): self.continuation(arm, 'before_state_commit')

    def test_cpu_both_arms_keep_committed_episode_exactly_with_offline_observer(self):
        for arm in ('control', 'anchor'):
            with self.subTest(arm=arm): self.continuation(arm, 'after_state_commit', observe=True)

    def test_state_zero_identity_is_saved_before_updates_and_lost_committed_state_is_not_reset(self):
        train = function(self, 'train_arm')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-state0-') as tmp:
            _, output, reg = register_case(self, tmp)
            for arm in ('control', 'anchor'):
                class Stop(RuntimeError): pass
                def hook(stage, episode):
                    if stage == 'after_state_commit' and episode == 0: raise Stop()
                with self.assertRaises(Stop): train(output, arm, commit_hook=hook)
                state = read_state(output, arm)
                self.assertEqual(state['episode'], 0)
                self.assertEqual(state['optimizer']['state'], {})
                self.assertIn(reg['registration_digest'], str(state['provenance']))
                self.assertIn(S2_SHA, str(state['provenance']))
                assert_same(self, state['model'], torch.load(ROOT / S2_REL, weights_only=True, map_location='cpu'))
                train(output, arm, resume=True)
                path = output / 'arms' / arm / 'training_state.pt'; path.unlink()
                committed = (output / 'arms' / arm / 'completed.json').read_bytes()
                with self.assertRaises(ValueError): train(output, arm, resume=True)
                self.assertEqual((output / 'arms' / arm / 'completed.json').read_bytes(), committed)

    def test_candidate_inventory_waits_for_both_arms_and_rejects_checkpoint_or_completion_drift(self):
        train = function(self, 'train_arm'); candidates = function(self, 'candidates_for_run')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-models-') as tmp:
            _, output, reg = register_case(self, tmp)
            train(output, 'control')
            with self.assertRaises(ValueError): candidates(output, reg)
            train(output, 'anchor'); models = candidates(output, reg)
            self.assertEqual(len({m['sha256'] for m in models}), len(models))
            manifest = read_json(output / 'model_manifest.json')
            self.assertEqual(set(manifest['endpoints']), {'baseline', 'control500', 'anchor500'})
            self.assertIn('aliases', manifest)
            manifest_digest = manifest.get('model_manifest_digest', manifest.get('digest'))
            self.assertRegex(manifest_digest, r'^[0-9a-f]{64}$')
            for arm in ('control', 'anchor'):
                endpoint = manifest['endpoints'][arm + '500']
                if isinstance(endpoint, dict): self.assertEqual(endpoint['episode'], 4)
            path = output / 'arms/anchor/checkpoints/4.pt'
            with path.open('ab') as f: f.write(b'drift')
            with self.assertRaises(ValueError): candidates(output, reg)


@unittest.skipUnless(os.environ.get('S2_KL_ACCEPTANCE_CUDA') == '1',
                     'Actual CUDA exact resume is opt-in: S2_KL_ACCEPTANCE_CUDA=1')
class CudaExactResumeTests(unittest.TestCase):
    def test_actual_cuda_both_arms_before_and_after_commit(self):
        self.assertTrue(torch.cuda.is_available(), 'CUDA acceptance explicitly requested but actual CUDA is unavailable')
        helper = ExactResumeTests()
        helper.device = 'cuda:0'
        for arm in ('control', 'anchor'):
            for stage in ('before_state_commit', 'after_state_commit'):
                with self.subTest(arm=arm, stage=stage): helper.continuation(arm, stage, observe=(stage=='after_state_commit'))


if __name__ == '__main__': unittest.main()
