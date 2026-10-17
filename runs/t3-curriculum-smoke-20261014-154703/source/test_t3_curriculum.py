"""Stage gates, split isolation, RNG purity and continuous real optimizer updates."""
from copy import deepcopy
import json
import pickle
import threading
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import torch

import run_t3_curriculum as runner
from bomberman_env import Bomb


def probe_result(stage, success=60, suicide=0, games=None):
    return {'games': len(runner.skill_cases(stage)) * 2 if games is None else games,
            'owned_kill_success': {'percent': success}, 'suicide': {'percent': suicide}, 'records': []}


class CurriculumChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def controller(self, smoke=False):
        return runner.Curriculum('curriculum', threading.Event(), {1: 0, 2: 0, 3: 0}, smoke)

    def test_gate_requires_two_complete_passes_and_minimum_age(self):
        c = self.controller()
        for episode in (1, 50, 99, 101, 149):
            self.assertFalse(c.probe_due(episode))
        self.assertTrue(c.probe_due(100))
        self.assertTrue(c.record_probe(100, probe_result(1)))
        self.assertEqual(c.stage, 1)
        self.assertTrue(c.record_probe(150, probe_result(1)))
        self.assertEqual((c.stage, c.stage_start), (2, 151))
        self.assertEqual(c.events[0]['age'], 150)
        self.assertFalse(c.probe_due(249))
        self.assertTrue(c.probe_due(250))

    def test_failed_probe_resets_streak_and_cap_stops_incomplete_stage(self):
        c = self.controller()
        c.record_probe(100, probe_result(1))
        self.assertFalse(c.record_probe(150, probe_result(1, success=59.9)))
        self.assertEqual(c.streak, 0)
        c.record_probe(200, probe_result(1))
        self.assertEqual(c.stage, 1)
        self.assertTrue(c.is_set())
        self.assertFalse(c.stop.is_set(), 'Skill failure must leave diagnostic evaluation possible')
        self.assertEqual(c.failure['reason'], 'skill_gate_not_met_within_200_episodes')

    def test_passes_at_150_and_200_can_advance_at_cap(self):
        c = self.controller()
        c.record_probe(100, probe_result(1, success=0))
        c.record_probe(150, probe_result(1))
        c.record_probe(200, probe_result(1))
        self.assertEqual(c.stage, 2)
        self.assertFalse(c.is_set())

    def test_suicide_guard_and_missing_games_reject_a_pass(self):
        for result in (probe_result(1, success=100, suicide=0.1), probe_result(1, games=1)):
            c = self.controller()
            self.assertFalse(c.record_probe(100, result))
            self.assertEqual(c.streak, 0)

    def test_cancelled_probe_never_advances_or_claims_skill_failure(self):
        for smoke in (False, True):
            c = self.controller(smoke)
            c.stop.set()
            self.assertFalse(c.record_probe(200, None))
            self.assertFalse(c.events)
            self.assertIsNone(c.failure)

    def test_final_stage_becomes_normal_rule_transfer(self):
        c = self.controller()
        c.stage, c.stage_start = 3, 301
        c.record_probe(400, probe_result(3))
        c.record_probe(450, probe_result(3))
        self.assertEqual((c.stage, c.stage_start), (4, 451))
        self.assertFalse(c.probe_due(600))
        env = runner.BaseEnv(board_size=9, num_agents=2, seed=runner.TRAIN_SEED)
        c.episode = 450
        obs = c.reset(env)
        self.assertEqual(c.setup['stage'], 'C4_transfer')
        self.assertEqual((c.setup['layout'], c.setup['opponent'], c.setup['wood_density']), ('random', 'rule', 0.7))
        self.assertEqual(obs[0].shape, (12, 9, 9))

    def test_scenario_reset_returns_current_observation_not_old_random_board(self):
        c = self.controller()
        c.rng = Mock(random=Mock(return_value=1.0), integers=Mock(return_value=0))
        env = runner.BaseEnv(board_size=9, num_agents=2, max_steps=7, seed=0)
        for stage in (1, 2, 3):
            c.stage = stage
            for _ in range(2):
                env.bombs = [Bomb(1, 1, 0, 1, 2)]
                obs = c.reset(env)
                case = next(x for x in c.train_cases if x['id'] == c.setup['scenario_id'])
                self.assertEqual((case['family'], case['split']), ('fork', 'train'))
                self.assertEqual(case['aid'], c.setup['learning_aid'])
                self.assertEqual(env.max_steps, 7)
                self.assertFalse(env.bombs or env.flames)
                self.assertEqual(env.step_count, 0)
                for actual, expected in zip(obs, runner.observations(env)):
                    np.testing.assert_array_equal(actual, expected)
                if stage >= 2:
                    self.assertEqual((env.agents[case['aid']].ammo, case['group']), (2, 'combo'))
                opponents = {1 - case['aid']: 'rule'}
                c.on_step(env, c.episode, case['aid'], [0, 0], opponents)
                opponent = opponents[1 - case['aid']]
                if stage == 3:
                    self.assertEqual(opponent, 'rule')
                else:
                    self.assertEqual(opponent.kind, 'static' if stage == 1 else 'evader')

    def test_control_reset_exactly_matches_normal_phaseC_seeded_maps(self):
        c = runner.Curriculum('control', threading.Event(), {})
        adapted = runner.BaseEnv(board_size=9, num_agents=2, seed=runner.TRAIN_SEED)
        base = deepcopy(adapted)
        for _ in range(4):
            actual, expected = c.reset(adapted), base.reset()
            self.assertEqual(pickle.dumps(adapted.__dict__), pickle.dumps(base.__dict__))
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(c.setup['opponent'], 'rule')
            self.assertFalse(c.probe_due(c.episode))

    def test_static_and_evader_opponents_use_existing_behavior(self):
        case = next(c for c in runner.scenarios(('train',)) if c['group'] == 'combo')
        env = runner.make_env(case)
        aid = 1 - case['aid']
        self.assertEqual(runner.ScriptedOpponent('static', aid).select_action(None, env, aid), 0)
        evader = runner.ScriptedOpponent('evader', aid)
        for _ in range(12):
            action = evader.select_action(None, env, aid)
            self.assertIn(action, range(5))
            env.step([0, action])
        self.assertFalse(env.bombs)

    def test_skill_probe_preserves_numpy_torch_cuda_weights_and_training_mode(self):
        net = runner.BaseNet(board_size=9).train()
        if torch.cuda.is_available():
            net.to('cuda')
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        before = pickle.dumps(np.random.get_state()), torch.get_rng_state().clone(), runner.model_digest(net)
        cuda_before = [torch.cuda.get_rng_state(i).clone() for i in devices]
        with patch.object(runner, 'play_case', wraps=runner.play_case) as play:
            result = runner.skill_probe(net, 2, threading.Event(), smoke=True)
        self.assertEqual(result['games'], 4)
        self.assertEqual(result['model_state_sha256'], before[2])
        self.assertTrue(all(call.args[0]['split'] == 'validation' and call.args[3] == 'evader'
                            for call in play.call_args_list))
        self.assertEqual(pickle.dumps(np.random.get_state()), before[0])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[1]))
        self.assertEqual(runner.model_digest(net), before[2])
        self.assertTrue(net.training)
        self.assertTrue(all(torch.equal(torch.cuda.get_rng_state(i), state) for i, state in zip(devices, cuda_before)))

    def test_cancelled_skill_probe_discards_partial_result_and_restores_rng(self):
        stop, net = threading.Event(), runner.BaseNet(board_size=9)
        stop.set()
        before = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
        self.assertIsNone(runner.skill_probe(net, 1, stop))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[0]))
        self.assertEqual(pickle.dumps(np.random.get_state()), before[1])

    def test_real_training_keeps_one_optimizer_scheduler_across_three_stages(self):
        directory = runner.ROOT / 'runs' / ('t3-curriculum-unit-' + uuid4().hex)
        optimizers, schedules, probes = [], [], []
        real_adam = runner.BaseAdam
        real_scheduler = runner.trainer.optim.lr_scheduler.CosineAnnealingLR
        def adam(*args, **kwargs):
            optimizer = real_adam(*args, **kwargs)
            optimizers.append(optimizer)
            return optimizer
        def scheduler(*args, **kwargs):
            obj = real_scheduler(*args, **kwargs)
            schedules.append(obj)
            return obj
        def probe(net, stage, *args):
            probes.append((stage, runner.model_digest(net), int(next(iter(optimizers[0].state.values()))['step'])))
            return probe_result(stage)
        def evaluate(policy, board, agents, count, seed, steps, *args):
            self.assertEqual((board, agents, count, seed, steps), (9, 2, 2, 21000, 8))
            return {'seed': seed, 'num_episodes': count, 'win_rate': 0, 'active_win_rate': 0,
                    'suicide_rate': 0, 'draw_rate': 100, 'avg_kills': 0, 'games': []}
        with patch.object(runner, 'BaseAdam', side_effect=adam), \
             patch.object(runner.trainer.optim.lr_scheduler, 'CosineAnnealingLR', side_effect=scheduler), \
             patch.object(runner, 'skill_probe', side_effect=probe), \
             patch('evaluate.evaluate_seeded', side_effect=evaluate), patch('builtins.print'):
            result = runner.train_arm('curriculum', directory, {1: 0, 2: 0, 3: 0}, threading.Event(), 'cpu', True)
        self.assertIs(runner.trainer.BombermanFFAEnv, runner.BaseEnv)
        self.assertIs(runner.trainer.ActorCriticNet, runner.BaseNet)
        self.assertIs(runner.trainer.optim.Adam, runner.BaseAdam)
        self.assertEqual((len(optimizers), len(schedules), schedules[0].T_max), (1, 1, 6))
        self.assertEqual([x[0] for x in probes], [1, 2, 3])
        self.assertEqual([x[2] for x in probes], [2, 4, 6])
        self.assertEqual(len({x[1] for x in probes}), 3, 'Current weights must change, not reload B each stage')
        self.assertEqual(result['optimizer_steps'], 6)
        self.assertTrue(result['training_complete'] and result['validation_complete'] and result['curriculum_stages_complete'])
        self.assertEqual(result['stage_episode_counts'], {'C1_static': 2, 'C2_evader': 2, 'C3_rule_mix': 2})
        rows = [json.loads(x) for x in (directory / 'metrics.jsonl').read_text().splitlines()]
        self.assertEqual([r['optimizer_steps'] for r in rows], list(range(1, 7)))
        self.assertEqual([r['learning_aid'] for r in rows], [0, 1, 0, 1, 0, 1])
        self.assertTrue(all(rows[i]['lr'] > rows[i + 1]['lr'] for i in range(5)))
        self.assertTrue((directory / 'final_phaseC.pt').is_file() and (directory / 'best_phaseC.pt').is_file())

    def test_existing_output_and_unavailable_cuda_are_rejected_before_training(self):
        with patch.object(runner, 'train_arm') as train:
            with self.assertRaises(FileExistsError):
                runner.run(runner.ROOT)
            with patch.object(runner.torch.cuda, 'is_available', return_value=False), patch('pathlib.Path.mkdir') as mkdir:
                with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
                    runner.run(runner.ROOT / ('runs/uncreated-' + uuid4().hex))
                mkdir.assert_not_called()
            train.assert_not_called()

    def test_changed_source_fails_preflight_before_probing_or_training(self):
        real = runner.digest
        with patch.object(runner, 'digest', side_effect=lambda p: 'changed' if p.name == 'bomberman_env.py' else real(p)):
            probe = Mock()
            with self.assertRaises(AssertionError):
                runner.preflight(probe, threading.Event())
            probe.assert_not_called()

    def test_incomplete_or_smoke_pair_cannot_enter_full_evaluation(self):
        full = {'training_complete': True, 'validation_complete': True, 'curriculum_stages_complete': True,
                'completed_episodes': 600, 'expected_episodes': 600}
        self.assertTrue(runner.full_pair_ready({'control': full, 'curriculum': full}))
        for changes in ({'completed_episodes': 599}, {'validation_complete': False},
                        {'curriculum_stages_complete': False}, {'expected_episodes': 6, 'completed_episodes': 6}):
            self.assertFalse(runner.full_pair_ready({'control': full, 'curriculum': {**full, **changes}}))
        self.assertFalse(runner.full_pair_ready({'control': full}))

    def test_final_random_evaluation_uses_paired_fresh_boards_and_both_corners(self):
        def evaluate(policy, board, agents, count, seed, steps, **kwargs):
            self.assertEqual((board, agents, count, seed, steps), (9, 2, 400, 31000, 500))
            return {'seed': seed, 'num_episodes': count, 'win_rate': 40, 'active_win_rate': 35,
                    'by_corner': {str(a): {'win_rate': 40, 'active_win_rate': 35} for a in (0, 1)},
                    'games': [{'seed': s, 'aid': a, 'active_win': 0} for s in range(31000, 31200) for a in (0, 1)]}
        with patch.object(runner, 'make_ac_fn') as load, patch.object(runner, 'evaluate_seeded', side_effect=evaluate), \
             patch.object(runner, 'digest', return_value='test-only'), patch.object(runner, 'dump'):
            results, complete = runner.evaluate_models(runner.ROOT / 'unused', threading.Event(), False)
        self.assertTrue(complete)
        self.assertEqual(load.call_count, 3)
        self.assertEqual(results['curriculum']['passive_win_rate'], 5)
        self.assertEqual(results['curriculum']['by_corner']['1']['passive_win_rate'], 5)
        self.assertEqual(runner.paired_active_bootstrap(results['curriculum'], results['control'])['board_pairs'], 200)

    def test_full_orchestration_requires_complete_pair_then_both_evaluations(self):
        output = runner.ROOT / 'runs' / ('t3-curriculum-flow-unit-' + uuid4().hex)
        full = {'training_complete': True, 'validation_complete': True, 'curriculum_stages_complete': True,
                'completed_episodes': 600, 'expected_episodes': 600}
        baseline = {s: probe_result(s) for s in (1, 2, 3)}
        results = {model: {'suicide_rate': 0, 'games': [{'seed': s, 'aid': a,
            'active_win': int(model == 'curriculum' and s % 5 == 0)} for s in range(31000, 31200) for a in (0, 1)]}
                   for model in ('original', 'control', 'curriculum')}
        order = []
        def arm(name, directory, guards, stop, device, smoke):
            self.assertEqual((guards, device, smoke), ({1: 0, 2: 0, 3: 0}, 'cpu', False))
            order.append(name)
            return full
        def evaluate(*args):
            order.append('random_evaluation')
            return results, True
        def heldout(*args):
            order.append('heldout')
            return {}, True
        before = runner.digest(runner.WARM_START)
        with patch.object(runner, 'preflight', return_value=baseline), \
             patch.object(runner, 'train_arm', side_effect=arm), \
             patch.object(runner, 'evaluate_models', side_effect=evaluate), \
             patch.object(runner, 'heldout_skills', side_effect=heldout), patch('builtins.print'), \
             patch('pathlib.Path.mkdir'), patch.object(runner.shutil, 'copyfile'), patch.object(runner, 'dump'), \
             patch.object(runner.shutil, 'make_archive') as archive:
            result = runner.run(output, device='cpu')
        self.assertEqual(order, ['control', 'curriculum', 'random_evaluation', 'heldout'])
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['full_protocol'] and result['eligible_for_replication'])
        self.assertEqual(result['decision'], 'replicate_before_replacing')
        self.assertEqual(before, runner.digest(runner.WARM_START))
        archive.assert_called_once_with(str(output) + '-send', 'zip', root_dir=output)

    def test_partial_pair_skips_heldout_and_marks_incomplete(self):
        output = runner.ROOT / 'runs' / ('t3-curriculum-partial-unit-' + uuid4().hex)
        partial = {'training_complete': False, 'validation_complete': False,
                   'curriculum_stages_complete': False, 'completed_episodes': 12, 'expected_episodes': 600}
        with patch.object(runner, 'preflight', return_value={s: probe_result(s) for s in (1, 2, 3)}), \
             patch.object(runner, 'train_arm', return_value=partial) as train, \
             patch.object(runner, 'evaluate_models') as evaluate, patch.object(runner, 'heldout_skills') as heldout, \
             patch('builtins.print'), patch('pathlib.Path.mkdir'), patch.object(runner.shutil, 'copyfile'), \
             patch.object(runner, 'dump'), patch.object(runner.shutil, 'make_archive'):
            result = runner.run(output, device='cpu')
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse(result['eligible_for_replication'])
        self.assertEqual(result['decision'], 'retain_original')
        self.assertEqual(train.call_count, 1)
        evaluate.assert_not_called()
        heldout.assert_not_called()

    def test_stage_failure_evaluates_equal_budget_only_without_unlocking_test(self):
        full = {'training_complete': True, 'validation_complete': True, 'curriculum_stages_complete': True,
                'completed_episodes': 600, 'expected_episodes': 600}
        failed = {**full, 'training_complete': False, 'validation_complete': False,
                  'curriculum_stages_complete': False, 'completed_episodes': 350,
                  'stage_failure': {'stage': 'C2_evader', 'episode': 350}}
        with patch.object(runner, 'preflight', return_value={s: probe_result(s) for s in (1, 2, 3)}), \
             patch.object(runner, 'train_arm', side_effect=[full, failed]), \
             patch.object(runner, 'evaluate_models') as evaluate, patch.object(runner, 'heldout_skills') as heldout, \
             patch.object(runner, 'matched_budget_diagnostic', return_value=True) as diagnostic, \
             patch('builtins.print'), patch('pathlib.Path.mkdir'), patch.object(runner.shutil, 'copyfile'), \
             patch.object(runner, 'dump'), patch.object(runner.shutil, 'make_archive'):
            result = runner.run(runner.ROOT / ('runs/uncreated-' + uuid4().hex), device='cpu')
        self.assertEqual(diagnostic.call_args.args[1], 350)
        self.assertFalse(diagnostic.call_args.args[2].is_set())
        self.assertEqual((result['status'], result['reason']), ('incomplete', 'stage_gate_failed'))
        self.assertTrue(result['matched_budget_diagnostic_complete'])
        self.assertFalse(result['eligible_for_replication'])
        evaluate.assert_not_called()
        heldout.assert_not_called()

    def test_matched_budget_uses_validation_seed_and_current_final_weights(self):
        output = runner.ROOT / 'unused'
        def evaluate(policy, board, agents, count, seed, steps, **kwargs):
            self.assertEqual((count, seed, steps), (100, 21000, 500))
            return {'win_rate': 10, 'active_win_rate': 10, 'by_corner': {}}
        with patch.object(runner, 'make_ac_fn') as load, patch.object(runner, 'evaluate_seeded', side_effect=evaluate), \
             patch.object(runner, 'digest', return_value='test-only'), patch.object(runner, 'dump'), patch('builtins.print'):
            self.assertTrue(runner.matched_budget_diagnostic(output, 350, threading.Event()))
        self.assertEqual([x.args[0] for x in load.call_args_list],
            [str(output / 'control/budget_ep350.pt'), str(output / 'curriculum/final_phaseC.pt')])


if __name__ == '__main__':
    unittest.main()
