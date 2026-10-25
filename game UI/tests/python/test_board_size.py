"""Independent acceptance tests for match board-size selection only."""
from unittest.mock import patch

from test_acceptance import APIBase, paused, request, tick


class BoardSizeTests(APIBase):
    def assert_grid(self, frame, size, count=None):
        self.assertEqual(len(frame["grid"]), size)
        self.assertTrue(all(len(row) == size for row in frame["grid"]))
        if count is not None:
            self.assertEqual(len(frame["agents"]), count)
        for agent in frame["agents"]:
            self.assertTrue(0 <= agent["row"] < size and 0 <= agent["col"] < size)

    def test_omitted_size_remains_eleven(self):
        for mode in ("play", "watch"):
            with self.subTest(mode=mode):
                _, frame = self.create(mode=mode)
                self.assert_grid(frame, 11, 3)

    def test_explicit_sizes_support_existing_modes_and_agent_counts(self):
        for size in (9, 11):
            for mode, counts in (("play", (3, 4)), ("watch", (2, 3, 4))):
                for count in counts:
                    with self.subTest(size=size, mode=mode, count=count):
                        sid, frame = self.create(board_size=size, mode=mode, num_agents=count)
                        self.assert_grid(frame, size, count)
                        with self.ws(sid) as ws:
                            self.command(ws, "step")
                            frame = self.state(sid)
                            self.assertEqual(tick(frame), 1, "Real inference and engine must step")
                            self.assert_grid(frame, size, count)
                            self.assertIsNone(frame.get("error"))

    def test_nine_step_and_reset_preserve_size_in_both_modes(self):
        for mode in ("play", "watch"):
            with self.subTest(mode=mode):
                sid, first = self.create(board_size=9, mode=mode)
                with self.ws(sid) as ws:
                    self.command(ws, "step")
                    once = self.state(sid)
                    self.assertEqual(tick(once), 1)
                    self.assert_grid(once, 9)
                    self.command(ws, "reset")
                    reset = self.state(sid)
                    self.assertEqual(tick(reset), 0)
                    self.assertTrue(paused(reset))
                    self.assertNotEqual(reset["match_id"], first["match_id"])
                    self.assert_grid(reset, 9)
                    self.command(ws, "step")
                    self.assertEqual(tick(self.state(sid)), 1)
                    self.assert_grid(self.state(sid), 9)

    def test_only_integer_nine_and_eleven_are_accepted(self):
        base = dict(mode="watch", model_id=self.model(), num_agents=2, max_steps=8)
        invalid = (0, -9, 7, 8, 10, 13, 9.0, 11.0, "9", "11", True, False, None, [], {})
        for value in invalid:
            with self.subTest(board_size=value, kind=type(value).__name__):
                code, data = request("POST", "/api/sessions", dict(base, board_size=value))
                if 200 <= code < 300:
                    self.sessions.append(data["session_id"])
                self.assertTrue(400 <= code < 500, f"Reject invalid board_size {value!r}: {code}: {data}")

    def test_valid_board_size_preserves_model_and_configuration_validation(self):
        # Establish that this field is accepted, so an extra-field error cannot mask a guard regression.
        self.create(board_size=9)
        base = dict(mode="play", model_id=self.model(), num_agents=3, max_steps=8, board_size=9)
        invalid = ({"model_id": "missing-model"}, {"model_id": "../checkpoints/best_phaseE_4a.pt"},
                   {"model_id": ""}, {"num_agents": 2}, {"num_agents": 5}, {"speed": 0},
                   {"speed": 61}, {"max_steps": 0}, {"mode": "unknown"}, {"unexpected": 1})
        for change in invalid:
            with self.subTest(change=change):
                code, data = request("POST", "/api/sessions", dict(base, **change))
                if 200 <= code < 300:
                    self.sessions.append(data["session_id"])
                self.assertTrue(400 <= code < 500, f"Existing validation must remain: {change}: {code}: {data}")

    def test_real_selected_model_and_engine_use_the_requested_size(self):
        from bridge.models import load_policy
        from bridge.protocol import SessionConfig
        from bridge.sessions import GameSession

        model_id = self.model()
        for size in (9, 11):
            with self.subTest(size=size):
                try:
                    config = SessionConfig(mode="watch", model_id=model_id, num_agents=2,
                                           board_size=size, max_steps=8, seed=912789)
                except ValueError as exc:
                    self.fail(f"SessionConfig must accept integer board_size={size}: {exc}")
                # Observe the existing model-loading boundary, while retaining real checkpoint loading.
                with patch("bridge.sessions.load_policy", wraps=load_policy) as loading:
                    session = GameSession(config)
                try:
                    self.assertEqual(loading.call_count, 1)
                    args, kwargs = loading.call_args
                    import inspect
                    loaded = inspect.signature(load_policy).bind(*args, **kwargs).arguments
                    self.assertEqual(loaded["model_id"], model_id)
                    self.assertEqual(loaded["board_size"], size)
                    self.assertEqual(session.policy.net.board_size, size)
                    self.assertEqual(session.env.board_size, size)
                    session.command({"command": "step", "command_id": "size-inference"})
                    self.assertEqual(tick(session.state()), 1)
                    self.assert_grid(session.state(), size, 2)
                finally:
                    session.close()
