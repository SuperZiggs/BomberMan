"""R0 reward arithmetic, unchanged learner updates, observation purity and runner guards."""
from copy import deepcopy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import pickle
import threading
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import torch

import audit_r0 as audit
import train_actor_critic as trainer
from benchmark_t3 import observations
from bomberman_env import (ACTION_BOMB, ACTION_STOP, POWERUP_AMMO, POWERUP_RANGE,
    MAX_AMMO, MAX_BLAST_RANGE, WOODEN_WALL)
from evaluate import evaluate_seeded
from test_combat import arena, put_bomb
from legacy_test_support import legacy_train, strip_sd


class RewardChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def parts(self, env, actions, aid=0, was_alive=True):
        _, raw, _, info = env.step(actions)
        parts = trainer._reward_components(raw[aid], aid, info, env.agents[aid].kills, was_alive)
        self.assertAlmostEqual(sum(parts.values()), trainer._shaped_reward(raw[aid], aid, info,
            env.agents[aid].kills), places=12)
        return parts, info

    def test_useful_items_only_and_time_penalty(self):
        for kind, field, cap in ((POWERUP_AMMO, 'ammo', MAX_AMMO), (POWERUP_RANGE, 'blast_range', MAX_BLAST_RANGE)):
            for full in (False, True):
                env = arena(2)
                if full:
                    setattr(env.agents[0], field, cap)
                env.powerups[(1, 1)] = kind
                parts, _ = self.parts(env, [0, 0])
                self.assertEqual(parts['item'], 0 if full else .05)
                self.assertEqual(parts['time'], -.001)

    def test_delayed_owned_kill_active_win_and_wall_bonus(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 1, 5
        env.grid[3, 4] = WOODEN_WALL
        put_bomb(env, 1, 3, 0, timer=3, blast_range=2)
        put_bomb(env, 3, 3, 0, timer=8, blast_range=1)
        for _ in range(3):
            parts, info = self.parts(env, [0, 0])
        self.assertEqual(info['kills'], [(0, 1)])
        self.assertEqual(parts['owned_kill'], 4)
        self.assertEqual(parts['active_win'], 10)
        self.assertEqual(parts['wall'], .15)
        self.assertEqual(env.bombs, [])
        self.assertAlmostEqual(sum(parts.values()), 14.149, places=12)

    def test_passive_win_cancelled_even_after_token_bomb(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 1
        put_bomb(env, 1, 5, 1, timer=1, blast_range=1)
        parts, info = self.parts(env, [ACTION_BOMB, ACTION_STOP])
        self.assertIn(0, info['bombs_placed'])
        self.assertEqual(parts['passive_win_env'], 10)
        self.assertEqual(parts['passive_win_cancellation'], -10)
        self.assertEqual(parts['active_win'], 0)
        self.assertAlmostEqual(sum(parts.values()), -.001, places=12)

    def test_simultaneous_self_and_enemy_sources_deny_kill_and_wall_bonus(self):
        env = arena(2)
        for a, pos in zip(env.agents, ((1, 1), (1, 3))):
            a.row, a.col = pos
            put_bomb(env, *pos, a.agent_id, timer=1, blast_range=4)
        parts, info = self.parts(env, [0, 0])
        self.assertIn((0, 1), info['kills'])
        self.assertIn(0, info['self_blast_deaths'])
        self.assertEqual(parts['owned_kill'], 0)
        self.assertEqual(parts['death'], -1)
        self.assertEqual(parts['self_blast'], -1.5)
        self.assertEqual(sum(parts.values()), -2.5)
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 3
        env.grid[3, 4] = WOODEN_WALL
        put_bomb(env, 3, 3, 0, timer=1, blast_range=1)
        parts, info = self.parts(env, [0, 0])
        self.assertEqual(info['walls_destroyed'][0], 1)
        self.assertEqual(parts['wall'], 0)

    def test_multiple_owned_kills_and_timeout_have_exact_contributions(self):
        env = arena(3)
        for a, pos in zip(env.agents, ((5, 1), (1, 5), (3, 3))):
            a.row, a.col = pos
        put_bomb(env, 1, 3, 0, timer=1, blast_range=4)
        parts, _ = self.parts(env, [0, 0, 0])
        self.assertEqual(parts['owned_kill'], 8)
        self.assertEqual(parts['active_win'], 10)
        self.assertAlmostEqual(sum(parts.values()), 17.999, places=12)
        env = arena(2)
        env.max_steps = 1
        parts, info = self.parts(env, [0, 0])
        self.assertTrue(info['episode_done'])
        self.assertTrue(all(info['alive']))
        self.assertEqual(sum(parts.values()), -.001)

    def test_lingering_flame_and_post_death_events(self):
        env = arena(3)
        env.flames[(1, 1)] = 2
        env._flame_sources[(1, 1)] = {0: 2, 1: 2}
        env._flame_owners[(1, 1)] = 1
        parts, info = self.parts(env, [0, 0, 0])
        self.assertIn((1, 0), info['kills'])
        self.assertEqual(parts['self_blast'], -1.5)
        env = arena(3)
        env.agents[0].alive = False
        env.grid[3, 4] = WOODEN_WALL
        put_bomb(env, 3, 3, 0, timer=1, blast_range=1)
        parts, _ = self.parts(env, [0, 0, 0], was_alive=False)
        self.assertEqual(sum(parts.values()), 0)
        self.assertEqual(parts['time'], 0)
        self.assertEqual(parts['wall'], 0)

    def test_mismatch_nonfinite_and_helper_purity(self):
        info = {'kills': [], 'alive': [True, True], 'episode_done': False, 'items_collected': {}, 'walls_destroyed': {0: 2}}
        original = deepcopy(info)
        for raw in (0, float('nan')):
            with self.assertRaises(ValueError):
                trainer._reward_components(raw, 0, info, 0, True)
        parts = trainer._reward_components(-.001, 0, info, 0, True)
        self.assertEqual(parts['wall'], .3)
        self.assertEqual(info, original)

    def test_default_and_logged_training_bit_exact_to_before_source_cpu(self):
        spec = importlib.util.spec_from_file_location('before_r0',
            audit.ROOT / 'runs/r0-implementation-20261015/before/train_actor_critic.py')
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        outputs = []
        for fn, flag in ((old.train, None), (legacy_train, False), (legacy_train, True)):
            rows, actions = [], []
            with patch('torch.save'), patch('os.makedirs'), patch('builtins.print'):
                kwargs = {} if flag is None else {'reward_diagnostics': flag}
                net, rewards, wins = fn(episodes=2, board_size=9, max_steps=20, lr=1e-4, device='cpu',
                    seed=2026101504, resume_from=str(audit.B), validation_every=0, run_report=False,
                    on_episode=rows.append, on_step=lambda e, ep, aid, a, opp: actions.append((ep, aid, list(a))), **kwargs)
            if flag:
                for row in rows:
                    self.assertAlmostEqual(sum(row.pop('reward_components').values()), row['reward'], places=12)
                    self.assertLessEqual(row.pop('reward_component_max_error'), 1e-10)
            outputs.append((deepcopy(net.state_dict()), rewards, wins, strip_sd(rows), actions))
        for result in outputs[1:]:
            self.assertEqual(outputs[0][1:], result[1:])
            self.assertTrue(all(torch.equal(v, result[0][key]) for key, v in outputs[0][0].items()))

    def test_logged_ffa_stops_reward_at_learner_death(self):
        env = arena(3)
        env.max_steps = 4
        put_bomb(env, 1, 1, 1, timer=1, blast_range=1)
        rows = []
        with patch.object(trainer, 'BombermanFFAEnv', return_value=env), \
             patch.object(env, 'reset', side_effect=lambda: observations(env)), \
             patch.object(trainer.RuleBasedOpponent, 'select_action_env', return_value=0), \
             patch('torch.distributions.Categorical.sample', side_effect=lambda: torch.tensor([0])), \
             patch('torch.save'), patch('os.makedirs'), patch('builtins.print'):
            trainer.train(episodes=1, board_size=7, num_agents=3, max_steps=4, device='cpu',
                validation_every=0, run_report=False, reward_diagnostics=True, on_episode=rows.append)
        self.assertEqual(env.step_count, 4)
        self.assertEqual(rows[0]['learner_steps'], 1)
        self.assertEqual(rows[0]['reward_components']['time'], 0)
        self.assertEqual(rows[0]['reward'], -1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_diagnostics_cuda_smoke_and_cancelled_partial_trajectory(self):
        rows = []
        with patch('torch.save'), patch('os.makedirs'), patch('builtins.print'):
            net, _, _ = trainer.train(episodes=1, board_size=9, max_steps=4, device='cuda',
                seed=2026101504, resume_from=str(audit.B), validation_every=0, run_report=False,
                reward_diagnostics=True, on_episode=rows.append)
        self.assertTrue(all(torch.isfinite(v).all() for v in net.state_dict().values()))
        self.assertAlmostEqual(sum(rows[0]['reward_components'].values()), rows[0]['reward'], places=12)
        stop, rows = threading.Event(), []
        def cancel(env, ep, aid, actions, opponents):
            if env.step_count:
                stop.set()
        with patch('torch.save'), patch('os.makedirs'), patch('builtins.print'):
            _, rewards, _ = trainer.train(episodes=1, board_size=9, max_steps=4, device='cpu',
                seed=2026101504, resume_from=str(audit.B), validation_every=0, run_report=False,
                reward_diagnostics=True, on_episode=rows.append, stop_event=stop, on_step=cancel)
        self.assertEqual(rows, [])
        self.assertEqual(rewards, [])


class AuditChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_observation_and_rollouts_preserve_rng_state_weights_and_baseline_actions(self):
        net = trainer.ActorCriticNet(board_size=9)
        trainer.smart_load_weights(net, str(audit.B), 'cpu')
        weights = deepcopy(net.state_dict())
        stop = threading.Event()
        for aid in (0, 1):
            np_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
            seen = audit.play_game(net, 21000, aid, 20, stop)
            bare = audit.play_game(net, 21000, aid, 20, stop, observed=False)
            self.assertEqual({k: seen[0][k] for k in audit.GAME_FIELDS}, bare[0])
            np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
            self.assertEqual(np_state[2:], np.random.get_state()[2:])
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
            self.assertEqual(len(seen[1]) + 1, len(seen[2]))
            self.assertEqual(seen[0]['mask_violations'], 0)
        reference = evaluate_seeded(lambda e, obs, aid: trainer.SnapshotOpponent(net, 'cpu').select_action(obs[aid], e, aid),
            9, 2, 2, 21000, 20)
        for aid in (0, 1):
            result = audit.play_game(net, 21000, aid, 20, stop)[0]
            self.assertEqual({k: result[k] for k in audit.GAME_FIELDS}, strip_sd(reference['games'][aid]))
        self.assertTrue(all(torch.equal(v, net.state_dict()[key]) for key, v in weights.items()))
        env = arena(2)
        before = pickle.dumps(env.__dict__)
        audit.observe(env, 0, trainer.get_action_mask(env, 0), ACTION_BOMB)
        self.assertEqual(before, pickle.dumps(env.__dict__))

    def test_empty_groups_and_abs_shares_do_not_divide_by_net_reward(self):
        self.assertEqual(audit.describe([]), {'games': 0})
        net = trainer.ActorCriticNet(board_size=9)
        trainer.smart_load_weights(net, str(audit.B), 'cpu')
        row = audit.play_game(net, 21000, 0, 2, threading.Event())[0]
        summary = audit.describe([row])
        self.assertAlmostEqual(sum(summary['gross_abs_shares'].values()), 1)
        self.assertAlmostEqual(sum(summary['net_abs_shares'].values()), 1)
        row['reward_components'] = {k: 0. for k in audit.COMPONENTS}
        self.assertTrue(all(v is None for v in audit.describe([row])['gross_abs_shares'].values()))

    def test_validation_mismatch_and_run_input_guards(self):
        with self.assertRaises(ValueError):
            audit.run(audit.ROOT / 'unused', budget_minutes=0)
        with self.assertRaises(FileExistsError):
            audit.run(audit.ROOT)
        with patch.object(audit, 'digest', return_value='changed'):
            with self.assertRaises(ValueError):
                audit.run(audit.ROOT / ('runs/uncreated-r0-' + uuid4().hex))
        with self.assertRaises(AssertionError):
            audit.validation_parity([dict(win=0, active_win=0, draw=1, suicide=0)], {'win_rate': 100})

    def test_cancelled_runner_records_incomplete_without_optimizer_or_plots(self):
        saved = {}
        read_text = Path.read_text
        def current_runtime_fixture(path, *args, **kwargs):
            text = read_text(path, *args, **kwargs)
            if path == audit.PARENT / 'config.json':
                config = json.loads(text)
                for name in ('bomberman_env.py', 'rule_based_bot.py', 'evaluate.py'):
                    config['source_sha256'][name] = audit.digest(audit.ROOT / name)
                return json.dumps(config)
            return text
        with patch.object(audit, 'ActorCriticNet', side_effect=lambda **k: torch.nn.Linear(1, 1)), \
             patch.object(audit, 'smart_load_weights'), patch.object(audit, 'play_game', return_value=None), \
             patch.object(Path, 'read_text', new=current_runtime_fixture), \
             patch.object(Path, 'mkdir'), patch.object(Path, 'write_text'), \
             patch.object(audit.shutil, 'copy2'), patch.object(audit.shutil, 'make_archive'), \
             patch.object(gzip, 'open', return_value=io.StringIO()), patch.object(audit, 'plot_results') as plot, \
             patch.object(audit, 'dump', side_effect=lambda p, data: saved.update({p.name: data})), \
             patch.object(trainer.optim, 'Adam', side_effect=AssertionError('Audit must never train')), patch('builtins.print'):
            result = audit.run(audit.ROOT / ('runs/uncreated-r0-' + uuid4().hex), smoke=True)
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse(result['validation_parity'])
        self.assertEqual((result['games_completed'], result['training_updates']), (0, 0))
        self.assertEqual(result['decision'], 'retain_original')
        self.assertEqual(saved['config.json']['seed'], 21000)
        plot.assert_not_called()


if __name__ == '__main__':
    unittest.main()
