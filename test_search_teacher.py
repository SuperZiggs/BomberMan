import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import train_actor_critic as trainer
import training_state


def run(search_teacher, override=None, **kwargs):
    """One 1-step episode with the CNN sampling STOP (0); search returns `override` or the default."""
    rows, calls = [], []

    def fake_search(env, aid, mask, default):
        calls.append(default)
        return (default if override is None else override), {'skipped': False}
    with patch('train_actor_critic.get_action_mask', return_value=[True] * 6), \
         patch('kill_search.search_action', side_effect=fake_search), \
         patch('torch.distributions.Categorical.sample', return_value=torch.tensor([0])), \
         patch('train_actor_critic.torch.save'), patch('train_actor_critic.os.makedirs'), patch('builtins.print'):
        trainer.train(episodes=1, board_size=7, max_steps=1, device='cpu', run_report=False,
                      validation_every=0, search_teacher=search_teacher, on_episode=rows.append, **kwargs)
    return rows[0], calls


class SearchTeacherTests(unittest.TestCase):
    def test_strictly_better_search_line_becomes_a_combat_label(self):
        row, calls = run(True, override=5)
        self.assertEqual(calls, [0])
        self.assertGreater(row['search_loss'], 0)
        self.assertEqual(row['combat_loss'], 0)  # own mean: rare labels are not diluted by heuristic ones
        self.assertEqual(row['search'], {'queries': 1, 'labels': 1, 'bomb_labels': 1})

    def test_agreeing_search_adds_no_label_and_off_never_queries(self):
        row, _ = run(True)
        self.assertEqual(row['search']['labels'], 0)
        self.assertEqual(row['search_loss'], 0)
        row, calls = run(False, override=5)
        self.assertEqual(calls, [])
        self.assertNotIn('search', row)

    def test_resume_replays_sudden_death_curriculum(self):
        config = dict(episodes=4, board_size=9, max_steps=8, device='cpu', run_report=False, seed=5,
                      validation_every=0, sudden_death_probability=0.5)
        flags = lambda rows: [r['sudden_death_enabled'] for r in rows]
        with tempfile.TemporaryDirectory() as folder, patch('builtins.print'):
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
        self.assertEqual(flags(whole), flags(first) + flags(rest))
        self.assertEqual([r['reward'] for r in whole], [r['reward'] for r in first + rest])

    def test_new_recipe_settings_are_part_of_the_resume_provenance(self):
        state = {'sudden_death_probability', 'combat_loss_coef', 'teacher_version', 'search_teacher',
                 'late_game_attack_multiplier', 'target_active_win_rate'}
        with tempfile.TemporaryDirectory() as folder, patch('builtins.print'):
            trainer.train(episodes=1, board_size=7, max_steps=1, device='cpu', run_report=False,
                          validation_every=0, checkpoint_dir=folder, save_training_state=True)
            saved = training_state.load_training_state(Path(folder) / 'training_state.pt')
        self.assertTrue(state <= saved['provenance']['config'].keys())
        self.assertIn('kill_search.py', saved['provenance']['sources'])
        self.assertIn('search_loss_coef', saved['provenance']['config'])

    def test_ratio_objectives_refuse_search_labels(self):
        with self.assertRaises(ValueError):
            trainer.train(episodes=1, board_size=7, max_steps=1, device='cpu', run_report=False,
                          validation_every=0, search_teacher=True, policy_objective='ppo')


if __name__ == '__main__':
    unittest.main()
