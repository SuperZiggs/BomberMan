"""Move qualification, masked certificates, loss routing and unchanged local training."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmark_t3 import make_env, observations, scenarios
from bomberman_env import Bomb, _MOVE_DELTA
from combat_move_teacher import certify_combat_move
from train_actor_critic import get_action_mask, get_attack_action, train


class CombatMoveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def near_goal(self):
        case = next(c for c in scenarios(('validation',), smoke=True) if c['group'] == 'approach')
        env = make_env(case)
        env.agents[0].row, env.agents[0].col = 3, 2
        return env

    def test_positive_move_certificates_all_roles_rotations_preserve_rng_and_live_state(self):
        accepted = 0
        for case in scenarios(('validation',)):
            if case['group'] != 'approach':
                continue
            env, aid = make_env(case), case['aid']
            for tick in range(4):
                agent = env.agents[aid]
                mask = get_action_mask(env, aid)
                action = get_attack_action(env, aid, env._danger_map(), mask)
                before = pickle.dumps(env.__dict__), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
                cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
                result = certify_combat_move(env, aid, action, mask)
                self.assertEqual(pickle.dumps(env.__dict__), before[0])
                self.assertEqual(pickle.dumps(np.random.get_state()), before[1])
                self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all() if cuda else [])))
                if tick == 3:
                    self.assertTrue(result['accepted'], (case['id'], result))
                if result['accepted']:
                    accepted += 1
                    self.assertGreater(result['pressure_next'], 0)
                    self.assertGreater(result['pressure_gain'], 0)
                    branch = deepcopy(env)
                    for planned in result['path']:
                        self.assertTrue(get_action_mask(branch, aid)[planned])
                        actions = [0, 0]
                        actions[aid] = planned
                        _, _, _, info = branch.step(actions)
                        self.assertTrue(branch.agents[aid].alive)
                        if planned == 5:
                            self.assertIn(aid, info['bombs_placed'])
                    self.assertFalse(branch.bombs or branch.flames)
                    self.assertEqual(result['path'].count(5), 1)
                actions = [0, 0]
                actions[aid] = action
                env.step(actions)
        self.assertGreaterEqual(accepted, 16)

    def test_rejects_mask_collision_capacity_and_non_increasing_pressure(self):
        env = self.near_goal()
        mask = get_action_mask(env, 0)
        mask[4] = False
        self.assertEqual(certify_combat_move(env, 0, 4, mask)['reason'], 'not_allowed_move')
        self.assertEqual(certify_combat_move(env, 0, 5, [True] * 6)['reason'], 'not_allowed_move')
        env.agents[1].row, env.agents[1].col = 3, 3
        self.assertEqual(certify_combat_move(env, 0, 4, [True] * 6)['reason'], 'move_failed')
        env = self.near_goal()
        env.agents[0].bombs_placed = 1
        env.bombs = [Bomb(1, 2, 0, 7, 2)]
        self.assertEqual(certify_combat_move(env, 0, 4, get_action_mask(env, 0))['reason'], 'next_bomb_masked')
        env = self.near_goal()
        with patch('combat_move_teacher._bomb_escape_pressure', return_value=0):
            self.assertEqual(certify_combat_move(env, 0, 4, get_action_mask(env, 0))['reason'], 'no_pressure_increase')

    def test_greedy_failure_and_horizon_limit_fall_back_without_accepting(self):
        env = self.near_goal()
        with patch('combat_move_teacher._bfs_safe_cell', return_value=None):
            self.assertEqual(certify_combat_move(env, 0, 4, get_action_mask(env, 0))['reason'], 'escape_not_certified')
        env.bomb_timer = 40
        self.assertEqual(certify_combat_move(env, 0, 4, get_action_mask(env, 0))['reason'], 'horizon_limit')

    def test_loss_routing_changes_only_qualified_move_and_never_overrides_cnn(self):
        for mode, accepted in (('local', True), ('local_combat_move', True), ('local_combat_move', False)):
            rows, sampled = [], []
            certificate = {'accepted': accepted, 'reason': 'accepted' if accepted else 'next_bomb_masked',
                           'pressure_gain': 0.5, 'path': [4, 5, 1], 'milliseconds': 1.0}
            def on_step(env, ep, aid, actions, opponents):
                if env.step_count:
                    sampled.append(actions[aid])
            with patch('train_actor_critic.get_action_mask', return_value=[True] * 6), \
                 patch('train_actor_critic.get_attack_action', return_value=4), \
                 patch('train_actor_critic.certify_combat_move', return_value=certificate) as certifier, \
                 patch('torch.distributions.Categorical.sample', return_value=torch.tensor([0])), \
                 patch('train_actor_critic.torch.save'), patch('train_actor_critic.os.makedirs'), patch('builtins.print'):
                train(episodes=1, board_size=7, max_steps=1, device='cpu', run_report=False,
                      validation_every=0, attack_teacher=mode, on_step=on_step, on_episode=rows.append)
            promoted = mode == 'local_combat_move' and accepted
            self.assertEqual(sampled, [0])
            self.assertEqual(rows[0]['navigation_loss'] == 0, promoted)
            self.assertEqual(rows[0]['combat_loss'] > 0, promoted)
            self.assertEqual(rows[0]['combat_bombs'], 0)
            self.assertEqual(rows[0]['move_guidance']['labels'], 1)
            self.assertEqual(rows[0]['move_guidance']['accepted'], int(promoted))
            self.assertEqual(certifier.call_count, int(mode != 'local'))

    def test_default_local_matches_before_snapshot_weights_rewards_actions(self):
        root = Path(__file__).resolve().parent
        before = root / 'runs/combat-move-implementation-20261015/before/train_actor_critic.py'
        spec = importlib.util.spec_from_file_location('before_combat_move_trainer', before)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        kwargs = dict(phase='C', episodes=2, board_size=9, num_agents=2, max_steps=12,
            seed=2026101501, resume_from=str(root / 'runs/pressure-C-next/final_phaseC.pt'),
            device='cpu', validation_every=0, run_report=False)
        observed = []
        for fn in (old.train, train):
            rows, actions = [], []
            with patch('torch.save'), patch('os.makedirs'), patch('builtins.print'):
                net, rewards, wins = fn(**kwargs, on_episode=rows.append,
                    on_step=lambda env, ep, aid, a, opp: actions.append((ep, aid, list(a))))
            observed.append((net.state_dict(), rewards, wins, rows, actions))
        self.assertEqual(observed[0][1:3], observed[1][1:3])
        self.assertEqual(observed[0][4], observed[1][4])
        self.assertTrue(all(torch.equal(observed[0][0][k], v) for k, v in observed[1][0].items()))
        for previous, current in zip(observed[0][3], observed[1][3]):
            self.assertEqual(previous, {k: v for k, v in current.items() if k != 'move_guidance'})

    def test_real_qualification_trains_with_both_loss_groups_cpu_and_cuda(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            case = next(c for c in scenarios(('validation',), smoke=True) if c['group'] == 'approach')
            env, metrics = make_env(case), []
            sample = [torch.tensor([a], device=device) for a in (2, 2, 4, 4)]
            with patch('train_actor_critic.BombermanFFAEnv', return_value=env), \
                 patch.object(env, 'reset', side_effect=lambda: observations(env)), \
                 patch('train_actor_critic.RuleBasedOpponent.select_action_env', return_value=0), \
                 patch('torch.distributions.Categorical.sample', side_effect=sample), \
                 patch('train_actor_critic.torch.save'), patch('train_actor_critic.os.makedirs'), patch('builtins.print'):
                train(episodes=1, board_size=9, max_steps=4, device=device, seed=2026101501,
                      run_report=False, validation_every=0, attack_teacher='local_combat_move', on_episode=metrics.append)
            self.assertEqual(metrics[0]['move_guidance']['accepted'], 2)
            self.assertEqual(metrics[0]['move_guidance']['accepted_agreements'], 2)
            self.assertGreater(metrics[0]['combat_loss'], 0)
            self.assertGreater(metrics[0]['navigation_loss'], 0)


if __name__ == '__main__':
    unittest.main()
