"""Independent observable acceptance tests; never performs real training.

Requirements: .superpowers/sdd/temporal-plan-follower/test-author-packet.md.
Only fixed policy decisions and real deterministic environment transitions run.
"""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import fields, is_dataclass
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from bomberman_env import (
    ACTION_BOMB, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STOP,
    ACTION_UP, Bomb, _MOVE_DELTA,
)
from rule_based_bot import _survival_map
import temporal_trap_planner as planner
import train_actor_critic as training
from test_temporal_trap_planner import legal_mask, mirror_env
from test_temporal_trap_regressions import open_pair_arena


ROUTE = (ACTION_BOMB, ACTION_RIGHT, ACTION_DOWN, ACTION_BOMB)


def exposed_actions(case, plan):
    """Discover the observable action sequence without choosing an API spelling."""
    names = {field.name for field in fields(plan)} if is_dataclass(plan) else set()
    names.update(name for name in dir(plan) if not name.startswith("_"))
    candidates = []
    for name in sorted(names):
        value = getattr(plan, name)
        if (isinstance(value, (tuple, list)) and value
                and all(isinstance(action, (int, np.integer))
                        and 0 <= int(action) < 6 for action in value)):
            candidates.append(tuple(int(action) for action in value))
    case.assertTrue(
        candidates,
        "Selected temporal plan must expose its nonempty ordered action sequence; "
        "first_action and placement positions cannot teach the intervening ticks",
    )
    case.assertTrue(all(actions == candidates[0] for actions in candidates),
                    "Public action-sequence aliases must agree")
    return candidates[0]


def fixture_plan(actions=ROUTE):
    actions = tuple(actions)
    position = (3, 2)
    placements = []
    for action in actions:
        if action == ACTION_BOMB:
            placements.append(position)
        elif action in _MOVE_DELTA:
            delta = _MOVE_DELTA[action]
            position = (position[0] + delta[0], position[1] + delta[1])
    return SimpleNamespace(
        first_action=actions[0], bomb_positions=tuple(placements),
        pressure_gain=0.5, completion_tick=len(actions) - 1,
        escape_margin=8, predicts_elimination=False,
        actions=actions, action_sequence=actions, sequence=actions,
        planned_actions=actions,
    )


class PlanSequenceAcceptance(unittest.TestCase):
    def selected(self, env, *, move_first=False, max_bombs=2):
        mask = legal_mask(env, 0)
        if move_first:
            mask = [False] * 6
            mask[ACTION_RIGHT] = True
        plan = planner.find_temporal_trap_plan(
            env, 0, env._danger_map(), mask,
            planner.TemporalTrapConfig(
                horizon=8 if max_bombs == 2 else 6,
                max_bombs=max_bombs, min_pressure_gain=0.01,
            ),
        )
        self.assertIsNotNone(plan, "Known safe deterministic arena must select a plan")
        return plan

    def assert_replays_placements(self, env, plan, actions):
        replay = deepcopy(env)
        actual_positions = []
        for action in actions:
            before = (replay.agents[0].row, replay.agents[0].col)
            self.assertTrue(legal_mask(replay, 0)[action],
                            "Every exposed action must be legal on its actual tick")
            _, _, _, info = replay.step([action, ACTION_STOP])
            self.assertTrue(replay.agents[0].alive, "Plan must survive its prefix")
            if action == ACTION_BOMB:
                self.assertIn(0, info["bombs_placed"])
                actual_positions.append(before)
        self.assertEqual(tuple(actual_positions), plan.bomb_positions)
        safe, _, _ = _survival_map(
            replay, replay.agents[0].row, replay.agents[0].col,
            safety_margin=0,
        )
        self.assertTrue(safe[0, replay.agents[0].row, replay.agents[0].col],
                        "A safe continuation must remain after the final placement")

    def test_selected_pair_exposes_replayable_ordered_actions(self):
        env = open_pair_arena()
        plan = self.selected(env)
        self.assertEqual(len(plan.bomb_positions), 2)
        actions = exposed_actions(self, plan)
        self.assertEqual(actions.count(ACTION_BOMB), 2)
        self.assertEqual(plan.first_action, actions[0])
        self.assert_replays_placements(env, plan, actions)

    def test_move_first_pair_keeps_prefix_before_first_bomb(self):
        env = open_pair_arena()
        plan = self.selected(env, move_first=True)
        self.assertEqual(len(plan.bomb_positions), 2)
        actions = exposed_actions(self, plan)
        self.assertEqual(actions[0], ACTION_RIGHT)
        self.assertEqual(plan.first_action, actions[0])
        self.assertGreater(actions.index(ACTION_BOMB), 0)
        self.assertEqual(actions.count(ACTION_BOMB), 2)
        self.assert_replays_placements(env, plan, actions)

    def test_singleton_stops_at_its_only_placement(self):
        env = open_pair_arena(bomb_timer=5, enemy=(1, 3))
        env.agents[0].ammo = 1
        plan = self.selected(env, move_first=True, max_bombs=1)
        self.assertEqual(len(plan.bomb_positions), 1)
        actions = exposed_actions(self, plan)
        self.assertEqual(actions.count(ACTION_BOMB), 1)
        self.assertEqual(actions[-1], ACTION_BOMB)
        self.assertEqual(plan.first_action, actions[0])
        self.assert_replays_placements(env, plan, actions)


class _FixedPolicy(torch.nn.Module):
    """One scalar is enough for trainer bookkeeping; learning is disabled."""
    def __init__(self):
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.zeros(()))

    def forward(self, observation):
        return self.scalar.expand(1, 6), self.scalar.expand(1, 1)


def control_loop(*, samples=ROUTE, plan_actions=ROUTE, hazard=None,
                 masked_tick=None, blocked_tick=None, rejected_bomb_tick=None,
                 ring_tick=None, no_plan=False):
    """Drive production control flow with all learning and external writes disabled."""
    env = open_pair_arena(bomb_timer=12)
    env.max_steps = len(samples)
    observations = [env._get_obs(aid) for aid in range(env.num_agents)]
    metrics, labels, rewards, observed = [], [], [], []
    tick = [0]
    policy_samples = iter(samples)
    fixed = _FixedPolicy()
    parameters_before = {key: value.clone() for key, value in fixed.state_dict().items()}
    actual_step = env.step
    actual_cross_entropy = training.F.cross_entropy
    plan = None if no_plan else fixture_plan(plan_actions)
    planner_calls = [0]

    def propose(*args, **kwargs):
        planner_calls[0] += 1
        return plan if planner_calls[0] == 1 else None

    def danger(state):
        result = np.zeros_like(state.grid, dtype=np.int32)
        if tick[0] == 1 and hazard == "fire":
            result[state.agents[0].row, state.agents[0].col] = 1
        elif tick[0] == 1 and hazard == "destination":
            row, col = state.agents[0].row, state.agents[0].col
            result[row, col + 1] = 1
        return result

    def mask(state, aid):
        result = legal_mask(state, aid)
        if tick[0] == masked_tick:
            result[ACTION_RIGHT] = False
        return result

    def step(actions):
        applied = list(actions)
        if tick[0] == blocked_tick or tick[0] == rejected_bomb_tick:
            applied[0] = ACTION_STOP
        transition = actual_step(applied)
        observed.append((tuple(actions), deepcopy(transition[3])))
        rewards.append(transition[1][0])
        tick[0] += 1
        return transition

    def capture_label(logits, target, *args, **kwargs):
        labels.append((tick[0], int(target.item())))
        return actual_cross_entropy(logits, target, *args, **kwargs)

    optimizer = Mock(name="disabled_optimizer")
    scheduler = Mock(name="disabled_scheduler")
    scheduler.get_last_lr.return_value = [0.0]
    with ExitStack() as stack:
        for manager in (
            patch.object(training, "BombermanFFAEnv", return_value=env),
            patch.object(env, "reset", return_value=observations),
            patch.object(env, "step", side_effect=step),
            patch.object(training, "ActorCriticNet", return_value=fixed),
            patch.object(training.optim, "Adam", return_value=optimizer),
            patch.object(training.optim.lr_scheduler, "CosineAnnealingLR", return_value=scheduler),
            patch.object(torch.Tensor, "backward"),
            patch.object(training.torch.nn.utils, "clip_grad_norm_"),
            patch.object(training.torch.distributions.Categorical, "sample",
                         side_effect=lambda: torch.tensor([next(policy_samples)])),
            patch.object(training, "RuleBasedOpponent",
                         side_effect=lambda aid: SimpleNamespace(
                             select_action_env=lambda state: ACTION_STOP)),
            patch.object(training, "get_action_mask", side_effect=mask),
            patch.object(training, "_build_danger_map", side_effect=danger),
            patch.object(training, "find_temporal_trap_plan", side_effect=propose),
            patch.object(training, "find_interception_action", return_value=None),
            patch.object(training, "_ring_safety_action", side_effect=lambda *a, **k:
                         ACTION_UP if tick[0] == ring_tick else None),
            patch.object(training, "_bfs_safe_cell", return_value=ACTION_LEFT),
            patch.object(training, "_sequence_attack_action", return_value=None),
            patch.object(training, "_pressure_attack_action", return_value=None),
            patch.object(training, "_bfs_toward_item", return_value=None),
            patch.object(training, "_bfs_toward_opponent", return_value=None),
            patch.object(training, "_bfs_toward_wood", return_value=None),
            patch.object(training.F, "cross_entropy", side_effect=capture_label),
            patch.object(training.torch, "save"),
            patch.object(training.os, "makedirs"),
            patch.object(training, "_print_report", side_effect=AssertionError("No evaluation allowed")),
            patch("builtins.print"),
            patch("builtins.open", side_effect=AssertionError("No artifact writes allowed")),
        ):
            stack.enter_context(manager)
        training.train(
            episodes=1, board_size=7, num_agents=2, max_steps=len(samples),
            log_every=100, device="cpu", seed=20261016,
            checkpoint_dir="unused-no-training-control-loop", run_report=False,
            validation_every=0, on_episode=metrics.append,
        )
    assert len(metrics) == 1, "The controlled episode must publish exactly once"
    assert all(torch.equal(value, parameters_before[key])
               for key, value in fixed.state_dict().items()), "No model training is allowed"
    return SimpleNamespace(metrics=metrics[0], labels=labels, observed=observed,
                           rewards=rewards, planner_calls=planner_calls[0])


class FollowerControlLoopAcceptance(unittest.TestCase):
    def test_matching_sample_supervises_whole_pending_sequence(self):
        result = control_loop()
        self.assertEqual(result.labels, list(enumerate(ROUTE)),
                         "Matching policy actions must retain the remaining temporal teacher labels")

    def test_sample_mismatch_cancels_before_following_tick(self):
        result = control_loop(samples=(ACTION_BOMB, ACTION_STOP, ACTION_STOP))
        self.assertEqual(result.labels, [(0, ACTION_BOMB), (1, ACTION_RIGHT)],
                         "Tick-one mismatch must cancel rather than advance to DOWN")

    def test_stale_transition_cancels_before_following_tick(self):
        result = control_loop(samples=(ACTION_BOMB, ACTION_RIGHT, ACTION_STOP), blocked_tick=1)
        self.assertEqual(result.labels, [(0, ACTION_BOMB), (1, ACTION_RIGHT)],
                         "A sampled movement that did not happen invalidates the pending route")

    def test_masked_next_action_cancels_pending_sequence(self):
        result = control_loop(samples=(ACTION_BOMB, ACTION_STOP, ACTION_STOP), masked_tick=1)
        self.assertEqual(result.labels, [(0, ACTION_BOMB)],
                         "A newly masked RIGHT must neither be taught nor advance")

    def test_unsafe_next_action_cancels_pending_sequence(self):
        result = control_loop(samples=(ACTION_BOMB, ACTION_STOP, ACTION_STOP), hazard="destination")
        self.assertEqual(result.labels, [(0, ACTION_BOMB)],
                         "A newly burning destination invalidates the pending movement")

    def test_fire_escape_preempts_pending_sequence(self):
        actions = (ACTION_RIGHT, ACTION_BOMB, ACTION_DOWN, ACTION_RIGHT, ACTION_BOMB)
        result = control_loop(samples=(ACTION_RIGHT, ACTION_LEFT, ACTION_STOP),
                              plan_actions=actions, hazard="fire")
        self.assertEqual(result.labels, [(0, ACTION_RIGHT), (1, ACTION_LEFT)],
                         "Urgent escape must supersede a pending BOMB even when physically legal")

    def test_ring_safety_preempts_pending_sequence(self):
        result = control_loop(samples=(ACTION_BOMB, ACTION_UP, ACTION_STOP), ring_tick=1)
        self.assertEqual(result.labels, [(0, ACTION_BOMB), (1, ACTION_UP)],
                         "Ring escape must supersede the pending RIGHT label")


class ObservedMetricsAcceptance(unittest.TestCase):
    def executed_bombs(self, payload):
        explicit = [key for key in payload
                    if "temporal" in key.lower() and "bomb" in key.lower()
                    and "execut" in key.lower()]
        key = explicit[0] if explicit else "temporal_trap_bombs"
        self.assertIn(key, payload, "Episode must report observed temporal bomb placements")
        self.assertIn(type(payload[key]), (int, float))
        return payload[key]

    def completed_pairs(self, payload):
        candidates = [key for key in payload
                      if ("temporal" in key.lower() or "sequence" in key.lower())
                      and ("complet" in key.lower())
                      and "predict" not in key.lower() and "tick" not in key.lower()]
        self.assertTrue(candidates,
                        "Episode must report observed completed two-bomb temporal sequences")
        for key in candidates:
            self.assertIn(type(payload[key]), (int, float))
        self.assertTrue(all(payload[key] == payload[candidates[0]] for key in candidates))
        return payload[candidates[0]]

    def test_proposed_but_unsampled_plan_counts_zero_executed_bombs(self):
        result = control_loop(samples=(ACTION_STOP,))
        self.assertEqual(result.observed[0][1]["bombs_placed"], [])
        self.assertEqual(self.executed_bombs(result.metrics), 0,
                         "A proposed BOMB label is not an observed placement")

    def test_proposed_but_unsampled_pair_counts_zero_completions(self):
        result = control_loop(samples=(ACTION_STOP,))
        self.assertEqual(self.completed_pairs(result.metrics), 0)

    def test_actual_pair_counts_two_executed_bombs_and_one_completion(self):
        result = control_loop()
        self.assertEqual(sum(0 in info["bombs_placed"] for _, info in result.observed), 2)
        self.assertEqual(self.executed_bombs(result.metrics), 2)
        self.assertEqual(self.completed_pairs(result.metrics), 1)

    def test_partial_pair_counts_one_bomb_and_no_completion(self):
        result = control_loop(samples=(ACTION_BOMB, ACTION_STOP, ACTION_STOP))
        self.assertEqual(self.executed_bombs(result.metrics), 1)
        self.assertEqual(self.completed_pairs(result.metrics), 0)

    def test_singleton_counts_one_bomb_and_no_pair_completion(self):
        actions = (ACTION_RIGHT, ACTION_BOMB)
        result = control_loop(samples=actions, plan_actions=actions)
        self.assertEqual(sum(0 in info["bombs_placed"] for _, info in result.observed), 1)
        self.assertEqual(self.executed_bombs(result.metrics), 1)
        self.assertEqual(self.completed_pairs(result.metrics), 0)

    def test_sampled_but_rejected_placement_counts_zero_bombs(self):
        result = control_loop(samples=(ACTION_BOMB,), rejected_bomb_tick=0)
        self.assertEqual(result.observed[0][1]["bombs_placed"], [])
        self.assertEqual(self.executed_bombs(result.metrics), 0,
                         "A sampled BOMB rejected by the transition was not placed")


class ForecastPreservationAcceptance(unittest.TestCase):
    def test_future_second_bomb_cannot_block_or_chain_react_early(self):
        env = open_pair_arena(bomb_timer=3)
        first = Bomb(3, 3, 0, 3, 2, placement_tick=1)
        second = Bomb(5, 3, 0, 3, 2, placement_tick=6)
        first_burns, first_blocked = env._forecast_blasts(first, min_horizon=11)
        pair_burns, pair_blocked = env._forecast_blasts([first, second], min_horizon=11)
        np.testing.assert_array_equal(pair_burns[:6], first_burns[:6])
        np.testing.assert_array_equal(pair_blocked[:6], first_blocked[:6])
        self.assertFalse(pair_blocked[0, 5, 3])
        self.assertTrue(first_burns[3, 5, 3])
        self.assertTrue(pair_blocked[6, 5, 3])
        self.assertTrue(pair_burns[8, 5, 3])

    def test_escape_margin_keeps_time_indexed_reachability(self):
        safe = np.ones((4, 5, 5), dtype=bool)
        blocked = np.ones_like(safe)
        start, bridge, refuge = (2, 1), (2, 2), (2, 3)
        for position in (start, bridge, refuge):
            blocked[:, position[0], position[1]] = False
        blocked[1:, bridge[0], bridge[1]] = True
        self.assertEqual(planner._reachable_escape_margin(safe, blocked, start), 1)


class PreservedContractAcceptance(unittest.TestCase):
    def test_plan_selection_preserves_live_state_rng_and_mirror_determinism(self):
        env = open_pair_arena()
        config = planner.TemporalTrapConfig(min_pressure_gain=0.01)
        state = pickle.dumps(env, protocol=5)
        rng = np.random.get_state()
        first = planner.find_temporal_trap_plan(env, 0, env._danger_map(), legal_mask(env, 0), config)
        second = planner.find_temporal_trap_plan(env, 0, env._danger_map(), legal_mask(env, 0), config)
        self.assertEqual(first, second)
        self.assertEqual(pickle.dumps(env, protocol=5), state)
        current_rng = np.random.get_state()
        self.assertEqual(rng[0], current_rng[0])
        np.testing.assert_array_equal(rng[1], current_rng[1])
        self.assertEqual(rng[2:], current_rng[2:])
        mirror = mirror_env(env)
        mirrored = planner.find_temporal_trap_plan(
            mirror, 0, mirror._danger_map(), legal_mask(mirror, 0), config)
        self.assertIsNotNone(first)
        self.assertIsNotNone(mirrored)
        self.assertEqual(mirrored.pressure_gain, first.pressure_gain)

    def test_reward_values_remain_independent_of_teacher_labels(self):
        suicide = {"kills": [], "alive": [False, True], "episode_done": True,
                   "self_blast_deaths": [0]}
        active = {"kills": [(0, 1)], "alive": [True, False], "episode_done": True}
        passive = {"kills": [], "alive": [True, False], "episode_done": True}
        self.assertEqual(training._combat_reward(-1.0, 0, suicide, 0), -2.5)
        self.assertEqual(training._combat_reward(10.0, 0, active, 1), 13.0)
        self.assertEqual(training._combat_reward(10.0, 0, passive, 0), 0.0)

    def test_action_mask_still_masks_walls_and_occupied_bomb_cell(self):
        env = open_pair_arena()
        env.agents[0].row, env.agents[0].col = 1, 1
        env.bombs.append(Bomb(1, 1, 0, 12, 2))
        env.agents[0].bombs_placed = 1
        mask = training.get_action_mask(env, 0)
        self.assertFalse(mask[ACTION_UP])
        self.assertFalse(mask[ACTION_LEFT])
        self.assertFalse(mask[ACTION_BOMB])
        self.assertTrue(mask[ACTION_RIGHT] or mask[ACTION_DOWN])


if __name__ == "__main__":
    unittest.main()
