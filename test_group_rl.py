import pickle
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from bomberman_env import ACTION_BOMB, ACTION_DOWN, ACTION_LEFT, ACTION_STOP, ACTION_UP
import group_rl
import train_actor_critic as trainer
import training_state
from test_cutoff_teacher import corridor


def holding_doorway():
    env = corridor()
    env.agents[0].row, env.agents[0].col = 2, 3  # learner already in the opponent's last refuge
    return env


class RolloutTests(unittest.TestCase):
    def test_outcome_is_credited_by_bomb_owner_and_leaves_the_game_untouched(self):
        env = holding_doorway()
        before, rng = pickle.dumps(env.__dict__), pickle.dumps(np.random.get_state())
        outcome, steps = group_rl.rollout(env, 0, lambda obs, mask: ACTION_STOP, seed=3)
        self.assertEqual(outcome, 1)            # holding the doorway: the learner's bomb kills
        self.assertTrue(steps and all(a == ACTION_STOP for _, _, a in steps))
        self.assertEqual(before, pickle.dumps(env.__dict__))
        self.assertEqual(rng, pickle.dumps(np.random.get_state()))
        outcome, _ = group_rl.rollout(env, 0, lambda obs, mask: ACTION_DOWN if mask[ACTION_DOWN] else ACTION_STOP, seed=3)
        self.assertNotEqual(outcome, 1)         # running away frees the exit: no learner kill

    def test_group_advantages_are_centred_and_zero_without_contrast(self):
        self.assertEqual(group_rl.group_advantages([1, 0, 0, 1]), [0.5, -0.5, -0.5, 0.5])
        self.assertEqual(group_rl.group_advantages([0, 0, 0]), [0.0, 0.0, 0.0])

    def test_group_loss_raises_the_probability_of_the_winning_action(self):
        torch.manual_seed(0)
        net = trainer.ActorCriticNet(in_channels=13, board_size=9)
        obs = holding_doorway()._get_obs(0)
        mask = [True] * 6
        groups = [[(1, [(obs, mask, ACTION_STOP)]), (0, [(obs, mask, ACTION_UP)])]]
        def p_stop():
            with torch.no_grad():
                return torch.softmax(net(trainer.obs_to_tensor(obs, 'cpu'))[0], -1)[0, ACTION_STOP].item()
        before = p_stop()
        loss, stats = group_rl.group_loss(net, groups, 'cpu')
        loss.backward()
        with torch.no_grad():
            for p in net.parameters():
                if p.grad is not None:  # the value head gets no policy gradient
                    p -= 0.1 * p.grad
        self.assertGreater(p_stop(), before)
        self.assertEqual((stats['groups'], stats['contrast_groups'], stats['rollouts']), (1, 1, 2))

    def test_first_step_loss_ignores_later_replay_steps(self):
        net = trainer.ActorCriticNet(in_channels=13, board_size=9)
        obs, mask = holding_doorway()._get_obs(0), [True] * 6
        groups = [[(1, [(obs, mask, ACTION_STOP), (obs, mask, ACTION_UP)]), (0, [(obs, mask, ACTION_UP)] * 3)]]
        _, stats = group_rl.group_loss(net, groups, 'cpu', first_only=True)
        self.assertEqual(stats['steps'], 2)
        self.assertEqual(group_rl.group_loss(net, groups, 'cpu')[1]['steps'], 5)


class NarrowTests(unittest.TestCase):
    def scores(self, scores):
        return patch('kill_search.search_action', return_value=(ACTION_STOP, {'skipped': False, 'scores': scores}))

    def test_keeps_only_states_with_one_or_two_killing_actions(self):
        env, mask = holding_doorway(), [True] * 6
        env.agents[1].row, env.agents[1].col = 2, 2  # opponent within 4
        with self.scores({ACTION_BOMB: 1.0, ACTION_STOP: 0.0, ACTION_UP: -1.0}):
            self.assertTrue(group_rl.narrow(env, 0, mask))
        with self.scores({ACTION_BOMB: 1.0, ACTION_STOP: 1.0, ACTION_UP: 0.0}):
            self.assertTrue(group_rl.narrow(env, 0, mask))
        with self.scores({ACTION_BOMB: 1.0, ACTION_STOP: 1.0, ACTION_UP: 1.0}):
            self.assertFalse(group_rl.narrow(env, 0, mask))   # wide window: already won, no signal
        with self.scores({ACTION_BOMB: 0.0, ACTION_STOP: 0.0}):
            self.assertFalse(group_rl.narrow(env, 0, mask))

    def test_needs_a_legal_bomb_and_a_near_opponent_before_searching(self):
        env = holding_doorway()
        with patch('kill_search.search_action') as search:
            self.assertFalse(group_rl.narrow(env, 0, [True] * 5 + [False]))   # BOMB illegal
            env.agents[1].row, env.agents[1].col = 7, 7
            self.assertFalse(group_rl.narrow(env, 0, [True] * 6))             # opponent too far
        search.assert_not_called()
        self.assertEqual(ACTION_BOMB, 5)


def train_rows(**kwargs):
    rows = []
    with patch('train_actor_critic.torch.save'), patch('train_actor_critic.os.makedirs'), patch('builtins.print'):
        trainer.train(episodes=2, board_size=9, max_steps=60, device='cpu', run_report=False, seed=11,
                      validation_every=0, on_episode=rows.append, **kwargs)
    return rows


class TrainerTests(unittest.TestCase):
    def test_zero_coefficient_leaves_training_bit_identical(self):
        # Force every candidate to count as decisive so the rollouts really run.
        with patch('group_rl.decisive', return_value=True):
            off = train_rows()
            torch.manual_seed(0)
            zero = train_rows(group_rl=True, group_coef=0.0)
            real, flips = group_rl.rollout, iter([1, 0] * 1000)
            with patch('group_rl.rollout', side_effect=lambda *a, **k: (next(flips), real(*a, **k)[1])):
                on = train_rows(group_rl=True, group_coef=1.0)  # alternating outcomes: every group has contrast
        self.assertGreater(sum(r['group']['contrast_groups'] for r in on), 0)
        self.assertEqual([r['reward'] for r in off], [r['reward'] for r in zero])
        self.assertEqual([r['loss'] for r in off], [r['loss'] for r in zero])
        self.assertGreater(sum(r['group']['states'] for r in zero), 0)
        self.assertNotIn('group', off[0])
        self.assertNotEqual([r['loss'] for r in off], [r['loss'] for r in on])

    def test_resume_replays_group_rollouts(self):
        config = dict(episodes=4, board_size=9, max_steps=60, device='cpu', run_report=False, seed=5,
                      validation_every=0, group_rl=True)
        with tempfile.TemporaryDirectory() as folder, patch('builtins.print'), \
             patch('group_rl.decisive', return_value=True):
            whole = []
            trainer.train(**config, checkpoint_dir=str(Path(folder) / 'whole'), on_episode=whole.append)
            first, rest, stop = [], [], threading.Event()
            def episode(row):
                first.append(row)
                if row['episode'] == 2:
                    stop.set()
            cut = Path(folder) / 'cut'
            trainer.train(**config, checkpoint_dir=str(cut), save_training_state=True, stop_event=stop,
                          on_episode=episode)
            trainer.train(**config, checkpoint_dir=str(cut), resume_training_state=str(cut / 'training_state.pt'),
                          on_episode=rest.append)
            saved = training_state.load_training_state(cut / 'training_state.pt')
        self.assertEqual([r['group'] for r in whole], [r['group'] for r in first + rest])
        self.assertEqual([r['loss'] for r in whole], [r['loss'] for r in first + rest])
        self.assertTrue({'group_rl', 'group_coef', 'group_k', 'group_horizon'} <= saved['provenance']['config'].keys())
        self.assertIn('group_rl.py', saved['provenance']['sources'])

    def test_buffer_replays_stored_states_in_later_episodes(self):
        found = iter([True, True] + [False] * 10000)  # two states in the first episode only
        with patch('group_rl.narrow', side_effect=lambda *a: next(found)):
            rows = train_rows(group_rl=True, group_select='narrow', group_buffer=10, group_first_step=True,
                              group_max_states=6, group_gap=1)
        self.assertEqual([(r['group']['new_states'], r['group']['states'], r['group']['buffer']) for r in rows],
                         [(2, 2, 2), (0, 2, 2)])

    def test_resume_restores_the_buffer(self):
        config = dict(episodes=4, board_size=9, max_steps=60, device='cpu', run_report=False, seed=5,
                      validation_every=0, group_rl=True, group_select='narrow', group_buffer=3,
                      group_first_step=True, group_gap=20)
        with tempfile.TemporaryDirectory() as folder, patch('builtins.print'),              patch('group_rl.narrow', return_value=True):
            whole = []
            trainer.train(**config, checkpoint_dir=str(Path(folder) / 'whole'), on_episode=whole.append)
            first, rest, stop = [], [], threading.Event()
            def episode(row):
                first.append(row)
                if row['episode'] == 2:
                    stop.set()
            cut = Path(folder) / 'cut'
            trainer.train(**config, checkpoint_dir=str(cut), save_training_state=True, stop_event=stop,
                          on_episode=episode)
            trainer.train(**config, checkpoint_dir=str(cut), resume_training_state=str(cut / 'training_state.pt'),
                          on_episode=rest.append)
        self.assertEqual([r['group'] for r in whole], [r['group'] for r in first + rest])
        self.assertEqual([r['loss'] for r in whole], [r['loss'] for r in first + rest])
        self.assertEqual(whole[-1]['group']['buffer'], 3)

    def test_zero_anchor_is_bit_identical_and_positive_anchor_changes_loss(self):
        off, zero, on = train_rows(), train_rows(anchor_coef=0.0), train_rows(anchor_coef=4.0)
        self.assertEqual([r['loss'] for r in off], [r['loss'] for r in zero])
        self.assertEqual([r['reward'] for r in off], [r['reward'] for r in on])  # same first-episode play
        self.assertNotIn('anchor_kl', off[0])
        self.assertAlmostEqual(on[0]['anchor_kl'], 0.0, places=6)  # episode 1: still the reference weights
        self.assertGreater(on[1]['anchor_kl'], 0.0)
        self.assertNotEqual([r['loss'] for r in off][1:], [r['loss'] for r in on][1:])

    def test_resume_keeps_the_warm_start_as_anchor(self):
        with tempfile.TemporaryDirectory() as folder, patch('builtins.print'):
            start = Path(folder) / 'start.pt'
            torch.save(trainer.ActorCriticNet(in_channels=13, board_size=9).state_dict(), start)
            config = dict(episodes=4, board_size=9, max_steps=60, device='cpu', run_report=False, seed=5,
                          validation_every=0, anchor_coef=2.0, resume_from=str(start))
            whole = []
            trainer.train(**config, checkpoint_dir=str(Path(folder) / 'whole'), on_episode=whole.append)
            first, rest, stop = [], [], threading.Event()
            def episode(row):
                first.append(row)
                if row['episode'] == 2:
                    stop.set()
            cut = Path(folder) / 'cut'
            trainer.train(**config, checkpoint_dir=str(cut), save_training_state=True, stop_event=stop,
                          on_episode=episode)
            trainer.train(**config, checkpoint_dir=str(cut), resume_training_state=str(cut / 'training_state.pt'),
                          on_episode=rest.append)
        self.assertEqual([r['anchor_kl'] for r in whole], [r['anchor_kl'] for r in first + rest])
        self.assertEqual([r['loss'] for r in whole], [r['loss'] for r in first + rest])

    def test_ratio_objectives_refuse_group_updates(self):
        with self.assertRaises(ValueError):
            trainer.train(episodes=1, board_size=7, max_steps=1, device='cpu', run_report=False,
                          validation_every=0, group_rl=True, policy_objective='ppo')
        with self.assertRaises(ValueError):
            trainer.train(episodes=1, board_size=7, max_steps=1, device='cpu', run_report=False,
                          validation_every=0, anchor_coef=1.0, policy_objective='ppo')


if __name__ == '__main__':
    unittest.main()
