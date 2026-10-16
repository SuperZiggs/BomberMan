from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import train_actor_critic as trainer
from sudden_death_experiment import SuddenDeathEnv, base_kwargs


class TrainerChannelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = self._dir.name

    def tearDown(self):
        self._dir.cleanup()

    def test_train_builds_net_with_env_channels(self):
        built = []
        base = trainer.ActorCriticNet

        def net(*args, **kwargs):
            built.append(base(*args, **kwargs))
            return built[-1]

        class RingEnv(SuddenDeathEnv):
            def __init__(self, **kwargs):
                super().__init__(**base_kwargs(kwargs), sudden_death=True, ring_channel=True)

        with patch.object(trainer, 'BombermanFFAEnv', RingEnv), \
             patch.object(trainer, 'ActorCriticNet', side_effect=net):
            trainer.train(phase='C', episodes=1, board_size=9, num_agents=2, max_steps=5,
                          device='cpu', run_report=False, validation_every=0, seed=1,
                          checkpoint_dir=self.tmp, log_every=1)
        self.assertEqual(built[0].conv[0].in_channels, 13)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_b_in_ring_env_picks_same_actions_as_without_channel(self):
        import evaluate
        from run_sudden_death_experiment import CHECKPOINT
        policy = evaluate.make_ac_fn(str(CHECKPOINT), 9, 'cpu')

        def play(ring):
            env = SuddenDeathEnv(board_size=9, num_agents=2, seed=40, sudden_death=True,
                                 sudden_death_start=20, sudden_death_interval=5, ring_channel=ring)
            obs, actions = env.reset(), []
            torch.manual_seed(7)
            for _ in range(40):
                action = policy(env, obs, 0)
                actions.append(action)
                obs, _, _, info = env.step([action, 0])
                if info['episode_done']:
                    break
            return actions

        self.assertEqual(play(False), play(True))

    def test_thirteen_channel_checkpoint_round_trip(self):
        import evaluate
        from run_sudden_death_experiment import CHECKPOINT
        net = trainer.ActorCriticNet(in_channels=13, board_size=9)
        trainer.smart_load_weights(net, str(CHECKPOINT), 'cpu')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ring.pt'
            torch.save(net.state_dict(), path)
            policy = evaluate.make_ac_fn(str(path), 9, 'cpu')
            env = SuddenDeathEnv(board_size=9, num_agents=2, seed=40, sudden_death=True, ring_channel=True)
            obs = env.reset()
            self.assertIn(policy(env, obs, 0), range(6))
            loaded = torch.load(path, weights_only=True)['conv.0.weight']
            self.assertEqual(loaded.shape[1], 13)
            self.assertFalse(torch.any(loaded[:, 12]))

    def test_wider_checkpoint_into_narrower_net_is_refused(self):
        from run_sudden_death_experiment import CHECKPOINT
        wide = trainer.ActorCriticNet(in_channels=13, board_size=9)
        trainer.smart_load_weights(wide, str(CHECKPOINT), 'cpu')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ring.pt'
            torch.save(wide.state_dict(), path)
            with self.assertRaises(ValueError):
                trainer.smart_load_weights(trainer.ActorCriticNet(in_channels=12, board_size=9), str(path), 'cpu')

    def test_b_logits_unchanged_in_thirteen_channel_net(self):
        from run_sudden_death_experiment import CHECKPOINT
        net12 = trainer.ActorCriticNet(in_channels=12, board_size=9)
        net13 = trainer.ActorCriticNet(in_channels=13, board_size=9)
        trainer.smart_load_weights(net12, str(CHECKPOINT), 'cpu')
        trainer.smart_load_weights(net13, str(CHECKPOINT), 'cpu')
        env = SuddenDeathEnv(board_size=9, num_agents=2, seed=5, sudden_death=True,
                             sudden_death_start=3, ring_channel=True)
        obs = env.reset()[0]
        self.assertTrue(obs[12].any())
        with torch.no_grad():
            a = net12(torch.as_tensor(obs[:12]).unsqueeze(0))
            b = net13(torch.as_tensor(obs).unsqueeze(0))
        torch.testing.assert_close(a, b, rtol=0, atol=0)


class RunnerGuardTests(unittest.TestCase):
    def test_refuses_existing_output_and_bad_budget(self):
        import run_sd_training_pilot as pilot
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                pilot.run(Path(folder), smoke=True)
        with self.assertRaises(ValueError):
            pilot.run(Path('runs/never-created-sd3'), budget_minutes=0)
        self.assertFalse(Path('runs/never-created-sd3').exists())

    def test_gate_requires_gain_ci_and_suicide(self):
        import run_sd_training_pilot as pilot

        def result(active, suicide=0):
            return {'suicide_rate': suicide, 'games': [
                {'seed': s, 'aid': a, 'active_win': int(s < active), 'win': 0, 'draw': 0, 'suicide': 0}
                for s in range(100) for a in (0, 1)]}
        base = result(30)
        self.assertTrue(pilot.gate({'original': base, 'sd_13ch': result(50)}, 'sd_13ch')[1])
        self.assertFalse(pilot.gate({'original': base, 'sd_13ch': result(32)}, 'sd_13ch')[1])
        self.assertFalse(pilot.gate({'original': base, 'sd_13ch': result(50, 1)}, 'sd_13ch')[1])

    def test_test_seed_is_fresh_and_distinct_from_smoke(self):
        import run_sd_training_pilot as pilot
        from run_combat_move_pilot import fresh_test_check
        if (pilot.ROOT / 'runs/sd3-full/test_opened.json').exists():
            # SD3 full consumed seed 40000; the guard must now refuse it.
            with self.assertRaises(ValueError):
                fresh_test_check(pilot.TEST_SEED)
        else:
            fresh_test_check(pilot.TEST_SEED)
        self.assertNotEqual(pilot.SMOKE_SEED, pilot.TEST_SEED)
        self.assertEqual(pilot.SD2['sudden_death_max_ring'], 2)


if __name__ == '__main__':
    unittest.main()
