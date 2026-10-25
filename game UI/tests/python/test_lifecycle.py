"""Focused regressions from the independent whole-feature review."""
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

UI = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(UI / ".python-deps"), str(UI), str(UI.parent)]
from bridge.protocol import SessionConfig, TrainingConfig
from bridge.sessions import GameSession
from bridge.training import TrainingJob, TrainingManager


class StopPolicy:
    def select_action(self, *args):
        return 0


class StopBot:
    def __init__(self, *args):
        pass

    def select_action(self, *args):
        return 0


class LifecycleTests(unittest.TestCase):
    def test_ring_warning_uses_engine_remaining_steps_once(self):
        from bomberman_env import BombermanFFAEnv
        from bridge.snapshots import capture_state
        env = BombermanFFAEnv(seed=912778, num_agents=4, sudden_death=True,
                              sudden_death_start=350, sudden_death_warning_steps=40)
        env.reset()
        for _ in range(309):
            env.step([0] * 4)
        self.assertIsNone(capture_state(env, ['A0', 'A1', 'A2', 'A3'])['ring']['remaining'])
        env.step([0] * 4)
        self.assertEqual(capture_state(env, ['A0', 'A1', 'A2', 'A3'])['ring']['remaining'], 40)
        for _ in range(39):
            env.step([0] * 4)
        warning = capture_state(env, ['A0', 'A1', 'A2', 'A3'])['ring']
        self.assertEqual(warning['remaining'], 1)
        self.assertIn([1, 1], warning['cells'])

    def make_session(self, policy=None, speed=12):
        with patch('bridge.sessions.load_policy', return_value=policy or StopPolicy()), patch('bridge.sessions.RuleBasedAgent', StopBot):
            return GameSession(SessionConfig(mode="play", model_id="test-id", num_agents=3,
                                            max_steps=8, seed=912780, speed=speed))

    def test_pause_gets_control_after_one_slow_tick(self):
        started = threading.Event()

        class SlowPolicy:
            def select_action(self, *args):
                started.set()
                time.sleep(.09)
                return 0

        session = self.make_session(SlowPolicy(), speed=60)
        try:
            session.command({"command": "resume", "command_id": "resume"})
            self.assertTrue(started.wait(2))
            before = time.monotonic()
            session.pause()
            elapsed = time.monotonic() - before
            self.assertLessEqual(session.state()["tick"], 2, "Pause must not wait through the rest of a slow match")
            self.assertLess(elapsed, .35, "Control must get a scheduling opportunity after the current slow tick")
        finally:
            session.close()

    def test_reset_preserves_input_sequence_high_water_mark(self):
        session = self.make_session()
        try:
            session.input({"seq": 10, "direction": "right"})
            session.command({"command": "reset", "command_id": "reset"})
            session.input({"seq": 9, "direction": "stop", "bomb": True})
            session.command({"command": "step", "command_id": "step"})
            self.assertEqual(session.state()["actions"][0], 0, "Pre-reset stale input must remain invalid")
        finally:
            session.close()

    def test_failed_worker_startup_is_terminal_and_releases_job_slot(self):
        import train_actor_critic  # Complete module initialization before testing worker failure.
        manager = TrainingManager()
        cfg = TrainingConfig(phase="C", episodes=1, num_agents=2, max_steps=1,
                             warm_start=True, model_id="test")
        failures = []
        with patch('threading.excepthook', side_effect=lambda event: failures.append(event.exc_value)):
            with patch('bridge.training.resolve_model', side_effect=[Path('allowed.pt'), ValueError('checkpoint disappeared')]):
                job = manager.start(cfg)
                job.thread.join(3)
            self.assertFalse(job.thread.is_alive())
            self.assertEqual(job.state()["status"], "error", "Every worker startup exception must become a terminal error")
            self.assertIn('checkpoint disappeared', job.state()["error"])
            self.assertEqual(failures, [], "Startup exceptions must not escape the worker boundary")

        def finished(worker):
            worker.status = "completed"

        with patch.object(TrainingJob, '_worker', finished):
            next_job = manager.start(TrainingConfig(phase="C", episodes=1, num_agents=2, max_steps=1))
            next_job.thread.join(2)
            self.assertEqual(next_job.state()["status"], "completed")
        manager.close()

    def test_thread_dispatch_failure_does_not_register_an_active_job(self):
        manager = TrainingManager()
        with patch('threading.Thread.start', side_effect=RuntimeError('thread unavailable')):
            with self.assertRaises(RuntimeError):
                manager.start(TrainingConfig(phase="C", episodes=1, num_agents=2, max_steps=1))
        self.assertEqual(manager.jobs, {}, "A thread that never started must not occupy an active-job slot")

    def test_corrupt_checkpoint_is_a_readable_recoverable_client_error(self):
        import asyncio
        import pickle
        from fastapi import HTTPException
        from bridge.app import create_session

        async def invoke(failure):
            caught = None
            try:
                with patch('bridge.sessions.load_policy', side_effect=failure('corrupt checkpoint')):
                    await create_session(SessionConfig(mode="play", model_id="valid-catalog-id", num_agents=3))
            except Exception as exc:
                caught = exc
            self.assertIsInstance(caught, HTTPException, "A corrupt weights error must be translated to a readable HTTP error")
            self.assertEqual(caught.status_code, 422)
            self.assertIn('corrupt checkpoint', caught.detail)
        for failure in (pickle.UnpicklingError, EOFError, IndexError):
            with self.subTest(failure=failure.__name__):
                asyncio.run(invoke(failure))


if __name__ == '__main__':
    unittest.main()
