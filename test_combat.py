"""Combat credit, anti-passivity rewards and attack targets: python test_combat.py."""
import unittest
import threading
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch, mock_open
import numpy as np
import torch

from bomberman_env import (
    BombermanFFAEnv, Bomb, ACTION_STOP, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_BOMB,
    RIGID_WALL, WOODEN_WALL, POWERUP_AMMO, POWERUP_RANGE,
    MAX_AMMO, MAX_BLAST_RANGE, OBS_CHANNELS,
)
from evaluate import run_match, tournament, evaluate_seeded
from rule_based_bot import (
    _build_danger_map, _bfs_safe_cell, _survival_map, _can_bomb_and_escape, RuleBasedAgent,
    _bomb_escape_pressure, _reachable_cells, _bfs_toward_opponent,
)
from train_actor_critic import (
    ActorCriticNet, SnapshotOpponent, smart_load_weights, train, _combat_reward, get_action_mask, get_attack_action, get_attack_action_v1,
    _supervision_losses,
)


def arena(num_agents=3):
    env = BombermanFFAEnv(board_size=7, num_agents=num_agents,
                          wooden_wall_density=0, powerup_prob=0, seed=7)
    env.reset()
    return env


def put_bomb(env, row, col, owner, timer=1, blast_range=2):
    env.bombs.append(Bomb(row, col, owner, timer, blast_range))
    env.agents[owner].bombs_placed += 1


class CombatChecks(unittest.TestCase):
    def test_safe_followup_traps_opponent_and_survives_simultaneous_counter_bomb(self):
        for learner in (0, 1):
            with self.subTest(learner=learner):
                env = arena(2)
                enemy = 1 - learner
                env.agents[learner].row, env.agents[learner].col = 3, 3
                env.agents[enemy].row, env.agents[enemy].col = 3, 5
                env.agents[learner].ammo = 2
                env.grid[1:6, 1:6] = RIGID_WALL
                for pos in [(3, 3), (3, 4), (3, 5), (3, 2), (3, 1), (2, 1), (1, 1), (1, 3), (2, 3)]:
                    env.grid[pos] = 0
                put_bomb(env, 1, 3, learner, timer=6, blast_range=2)
                danger = env._danger_map()
                self.assertEqual(danger[3, 3], 6)
                self.assertEqual(_bomb_escape_pressure(env, learner, 3, 3), 1)
                mask = get_action_mask(env, learner)
                self.assertTrue(mask[ACTION_BOMB])
                # v1 (pre-merge) follow-up rule; v2 escapes instead (test_phase_c_attack_750).
                self.assertEqual(get_attack_action_v1(env, learner, danger, mask), ACTION_BOMB)
                from rule_based_bot import _safe_action_mask
                self.assertFalse(_safe_action_mask(env, learner, robust=False)[ACTION_BOMB],
                                 'The opponent baseline must retain its original filter')
                actions = [ACTION_BOMB, ACTION_BOMB]
                env.step(actions)
                for action in [ACTION_LEFT, ACTION_LEFT, ACTION_UP, ACTION_STOP, ACTION_STOP]:
                    actions = [ACTION_STOP, ACTION_STOP]
                    actions[learner] = action
                    _, _, _, info = env.step(actions)
                    if info['episode_done']:
                        break
                self.assertTrue(env.agents[learner].alive)
                self.assertFalse(env.agents[enemy].alive)
                self.assertEqual(info['kills'], [(learner, enemy)])

    def test_followup_requires_a_trap_and_a_safe_placement_tick(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 3, 5
        env.agents[0].ammo = 2
        put_bomb(env, 1, 3, 0, timer=6, blast_range=2)
        self.assertLess(_bomb_escape_pressure(env, 0, 3, 3), 1)
        self.assertFalse(get_action_mask(env, 0)[ACTION_BOMB], 'An escapable opponent is not a finishing trap')
        env.bombs[0].timer = 1
        self.assertFalse(get_action_mask(env, 0)[ACTION_BOMB], 'Placing a bomb must not spend the only escape tick')

    def test_pressure_distinguishes_a_trap_from_an_open_arena_without_mutation(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 3, 5
        open_pressure = _bomb_escape_pressure(env, 0, 3, 3)
        self.assertGreater(open_pressure, 0)
        self.assertLess(open_pressure, 1)
        env.grid[1:6, 1:6] = RIGID_WALL
        for pos in [(3, 3), (3, 4), (3, 5), (2, 3), (1, 3), (1, 2), (1, 1)]:
            env.grid[pos] = 0
        state = deepcopy(env)
        self.assertEqual(_bomb_escape_pressure(env, 0, 3, 3), 1)
        mask = get_action_mask(env, 0)
        self.assertTrue(mask[ACTION_BOMB], 'The bomber must have its own escape route')
        self.assertEqual(get_attack_action(env, 0, env._danger_map(), mask), ACTION_BOMB)
        np.testing.assert_array_equal(env.grid, state.grid)
        self.assertEqual(env.bombs, state.bombs)
        self.assertEqual(env.agents, state.agents)

    def test_pressure_does_not_reward_an_enemy_already_doomed_by_visible_bombs(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 3, 5
        for pos in [(2, 5), (4, 5), (3, 4)]:
            env.grid[pos] = RIGID_WALL
        put_bomb(env, 3, 5, 1, timer=2, blast_range=1)
        self.assertEqual(_bomb_escape_pressure(env, 0, 3, 1), 0)
        env.agents[0].alive = False
        self.assertEqual(_bomb_escape_pressure(env, 0, 3, 1), 0)

    def test_teacher_can_plan_a_position_while_ammo_is_busy_without_bombing(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 3, 3
        env.agents[0].bombs_placed = env.agents[0].ammo
        mask = get_action_mask(env, 0)
        self.assertFalse(mask[ACTION_BOMB])
        target = get_attack_action(env, 0, env._danger_map(), mask)
        self.assertNotIn(target, (ACTION_BOMB, None))
        self.assertTrue(mask[target])

    def test_reachable_positions_match_real_movement_and_damage(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 1, 1
        env.agents[1].row, env.agents[1].col = 3, 5
        put_bomb(env, 3, 3, 0, timer=3, blast_range=2)
        burns, blocked = env._forecast_blasts()
        states = [env]
        for ticks in range(1, 4):
            next_states = []
            for state in states:
                for action in (ACTION_STOP, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
                    actual = deepcopy(state)
                    actual.step([ACTION_STOP, action])
                    if actual.agents[1].alive:
                        next_states.append(actual)
            states = next_states
            expected = {(s.agents[1].row, s.agents[1].col) for s in states}
            reached = _reachable_cells((3, 5), burns, blocked, ticks)
            self.assertEqual(set(zip(*np.nonzero(reached))), expected)

    def test_forecast_extension_preserves_wall_opening_and_expired_flames(self):
        env = arena(2)
        env.grid[3, 3] = WOODEN_WALL
        put_bomb(env, 3, 2, 0, timer=1, blast_range=2)
        burns, blocked = env._forecast_blasts()
        extended_burns, extended_blocked = env._forecast_blasts(min_horizon=9)
        np.testing.assert_array_equal(extended_burns[:len(burns)], burns)
        np.testing.assert_array_equal(extended_blocked[:len(blocked)], blocked)
        self.assertFalse(extended_burns[len(burns):].any())
        self.assertFalse(extended_blocked[2:, 3, 3].any())

    def test_combat_gradient_is_not_diluted_by_navigation_label_count(self):
        for count in (1, 100):
            combat = torch.tensor(1.0, requires_grad=True)
            navigation = torch.tensor(1.0, requires_grad=True)
            escape, bomb, move = _supervision_losses([], [combat.square()], [navigation.square()] * count)
            (0.5 * escape + 0.5 * bomb + 0.2 * move).backward()
            self.assertAlmostEqual(combat.grad.item(), 1.0)
            self.assertAlmostEqual(navigation.grad.item(), 0.4, places=5)
        self.assertEqual(_supervision_losses([], [], []), (0, 0, 0))

    def test_validation_keeps_warm_start_when_active_gain_increases_suicides(self):
        env = arena(2)
        env.max_steps = 1
        validations = [dict(active_win_rate=a, win_rate=a, suicide_rate=s,
                            draw_rate=85, avg_kills=a/100, games=[])
                       for a, s in [(10, 3), (11, 6), (12, 3)]]
        metrics = []
        with patch('train_actor_critic.BombermanFFAEnv', return_value=env), \
             patch('train_actor_critic.torch.save'), patch('train_actor_critic.os.makedirs'), \
             patch('builtins.open', mock_open()), patch('builtins.print'), \
             patch('evaluate.evaluate_seeded', side_effect=validations):
            train(episodes=2, board_size=7, max_steps=1, device='cpu', log_every=100,
                  checkpoint_dir='unused-test-checkpoints', run_report=False, on_episode=metrics.append,
                  validation_every=1, validation_episodes=100)
        self.assertEqual([m['best_episode'] for m in metrics], [0, 2])
        self.assertEqual([m['validation']['suicide_limit'] for m in metrics], [3, 3])

    def test_simultaneous_self_and_enemy_blast_credit_is_fair_at_both_corners(self):
        for learner in (0, 1):
            with self.subTest(learner=learner):
                env = BombermanFFAEnv(board_size=9, num_agents=2, wooden_wall_density=0)
                env.reset()
                env.agents[learner].row, env.agents[learner].col = 1, 3
                env.agents[learner].blast_range = 4
                env.agents[1-learner].row, env.agents[1-learner].col = 1, 1
                env.step([ACTION_BOMB, ACTION_BOMB])
                for action in [ACTION_DOWN, ACTION_DOWN, ACTION_LEFT] + [ACTION_STOP] * 4:
                    actions = [ACTION_STOP, ACTION_STOP]
                    actions[learner] = action
                    _, rewards, _, info = env.step(actions)
                    if info['episode_done']:
                        break
                self.assertTrue(env.agents[learner].alive)
                self.assertEqual(info['kills'], [(learner, 1-learner)])
                self.assertEqual(info['self_blast_deaths'], [1-learner])
                self.assertEqual(env.agents[learner].kills, 1)
                self.assertAlmostEqual(_combat_reward(rewards[learner], learner, info, 1), 13.999)

    def test_own_blast_death_stays_penalized_with_enemy_kill_credit(self):
        env = BombermanFFAEnv(board_size=9, num_agents=2, wooden_wall_density=0)
        env.reset()
        for agent, pos in zip(env.agents, [(1, 1), (1, 3)]):
            agent.row, agent.col = pos
            put_bomb(env, *pos, agent.agent_id, timer=1, blast_range=4)
        _, rewards, _, info = env.step([ACTION_STOP, ACTION_STOP])
        self.assertEqual(info['kills'], [(1, 0), (0, 1)])
        self.assertEqual(info['self_blast_deaths'], [0, 1])
        for aid in (0, 1):
            self.assertEqual(_combat_reward(rewards[aid], aid, info, 1), -2.5)

    def test_flame_refresh_does_not_extend_the_expired_owners_credit(self):
        env = BombermanFFAEnv(board_size=9, num_agents=3, wooden_wall_density=0)
        env.reset()
        for agent, pos in zip(env.agents, [(1, 7), (7, 7), (4, 3)]):
            agent.row, agent.col = pos
        put_bomb(env, 3, 1, 0, timer=1, blast_range=2)
        env.step([ACTION_STOP] * 3)
        put_bomb(env, 3, 5, 1, timer=1, blast_range=2)
        env.step([ACTION_STOP] * 3)
        self.assertEqual(env._flame_sources[(3, 3)], {1: 1})
        _, _, _, info = env.step([ACTION_STOP, ACTION_STOP, ACTION_UP])
        self.assertEqual(info['kills'], [(1, 2)])
        self.assertEqual(env._flame_sources, {})
        self.assertEqual(env._flame_owners, {})

    def test_evaluation_counts_own_blast_involvement_without_duplicate_deaths(self):
        env = BombermanFFAEnv(board_size=9, num_agents=2, wooden_wall_density=0)
        obs = env.reset()
        for agent, pos in zip(env.agents, [(1, 1), (1, 3)]):
            agent.row, agent.col = pos
            put_bomb(env, *pos, agent.agent_id, timer=1, blast_range=4)
        with patch.object(env, 'reset', return_value=obs):
            stats = run_match(env, [lambda *args: ACTION_STOP] * 2, num_episodes=1)
        self.assertEqual(stats['suicide_rates'], [100, 100])
        self.assertEqual(stats['active_wins'], [0, 0])
        self.assertEqual(stats['avg_kills'], [1, 1])

    def test_equal_refuges_do_not_escape_toward_a_closing_opponent(self):
        env = BombermanFFAEnv(board_size=9, num_agents=2, wooden_wall_density=0)
        env.reset()
        env.agents[0].row, env.agents[0].col = 5, 1
        env.agents[0].blast_range = 3
        env.agents[1].row, env.agents[1].col = 1, 1
        env.agents[1].blast_range = 4
        put_bomb(env, 5, 1, 0, timer=7, blast_range=3)
        put_bomb(env, 1, 1, 1, timer=7, blast_range=4)
        mask = get_action_mask(env, 0)
        self.assertFalse(mask[ACTION_UP], 'This escape meets the opponent at (3,1)')
        self.assertTrue(mask[ACTION_DOWN] or mask[ACTION_RIGHT])
        env.agents[0].row, env.agents[0].col = 1, 1
        env.agents[1].row, env.agents[1].col = 5, 1
        for bomb in env.bombs:
            bomb.owner = 1 - bomb.owner
        mask = get_action_mask(env, 0)
        self.assertFalse(mask[ACTION_DOWN], 'Apply the same separation rule after swapping positions')
        self.assertTrue(mask[ACTION_RIGHT])

    def test_incoming_bomb_keeps_multiple_safe_escape_directions(self):
        env = BombermanFFAEnv(board_size=9, num_agents=2, wooden_wall_density=0)
        env.reset()
        env.agents[0].row, env.agents[0].col = 5, 1
        env.agents[1].row, env.agents[1].col = 1, 3
        put_bomb(env, 5, 3, 1, timer=3, blast_range=3)
        mask = get_action_mask(env, 0)
        self.assertTrue(mask[ACTION_UP], 'Do not force retreat when dodging another agent bomb')
        self.assertTrue(mask[ACTION_DOWN])

    def test_attack_teacher_routes_via_allowed_moves_to_a_firing_position(self):
        env = BombermanFFAEnv(board_size=9, num_agents=2, wooden_wall_density=0)
        env.reset()
        env.agents[0].row, env.agents[0].col = 1, 1
        env.agents[1].row, env.agents[1].col = 1, 5
        danger = _build_danger_map(env)
        mask = get_action_mask(env, 0)
        mask[ACTION_RIGHT] = False
        self.assertEqual(get_attack_action(env, 0, danger, mask), ACTION_DOWN,
                         'Use the longer allowed route instead of dropping the attack label')
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 3, 4
        mask = get_action_mask(env, 0)
        mask[ACTION_BOMB] = False
        self.assertFalse(mask[ACTION_RIGHT])
        self.assertEqual(_bfs_toward_opponent(env, 0, 3, 3, danger, action_mask=mask), ACTION_LEFT,
                         'Do not route into the cell occupied by the opponent')
        target = get_attack_action(env, 0, danger, mask)
        self.assertNotIn(target, (ACTION_RIGHT, ACTION_BOMB, ACTION_STOP))
        self.assertTrue(mask[target])

    def test_margin_is_exactly_two_ticks_and_does_not_change_real_forecast(self):
        env = arena(2)
        put_bomb(env, 3, 3, 0, timer=8)
        actual, _ = env._forecast_blasts()
        _, _, padded = _survival_map(env, 1, 1, safety_margin=2)
        self.assertFalse(padded[4, 3, 3])
        self.assertTrue(padded[5, 3, 3])
        np.testing.assert_array_equal(env._forecast_blasts()[0], actual)

    def test_new_bomb_rejects_a_corridor_with_a_simultaneous_counter_bomb(self):
        env = BombermanFFAEnv(board_size=9, num_agents=2, wooden_wall_density=0)
        env.reset()
        env.agents[0].row, env.agents[0].col = 1, 1
        env.agents[0].blast_range = 3
        env.agents[1].row, env.agents[1].col = 1, 4
        env.agents[1].blast_range = 4
        put_bomb(env, 3, 5, 1, timer=4, blast_range=4)
        env.agents[1].ammo = 2
        self.assertFalse(_can_bomb_and_escape(env, 0, 1, 1, None))
        self.assertFalse(get_action_mask(env, 0)[ACTION_BOMB])
        from rule_based_bot import _safe_action_mask
        self.assertTrue(_safe_action_mask(env, 0, robust=False)[ACTION_BOMB],
                        'Preserve the fixed RuleBot baseline while hardening AC')

    def test_escape_prefers_a_longer_refuge_outside_counter_bomb_rays(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 4, 5
        env.grid[2, 3] = WOODEN_WALL
        env.grid[5, 2] = RIGID_WALL
        env.grid[4, 4] = 0
        put_bomb(env, 3, 5, 0, timer=8, blast_range=4)
        mask = get_action_mask(env, 0)
        self.assertFalse(mask[ACTION_DOWN], 'The nearest refuge is in a nearby opponent blast ray')
        self.assertTrue(mask[ACTION_LEFT], 'Keep the longer route in case the nearest exit closes')
        self.assertFalse(mask[ACTION_STOP])

    def test_proposed_placements_forecast_matches_actual_without_mutation(self):
        env = arena(2)
        placements = [Bomb(3, 1, 0, 3, 2), Bomb(3, 3, 1, 8, 2)]
        burns, _ = env._forecast_blasts(placements)
        self.assertEqual(env.bombs, [])
        self.assertEqual([b.timer for b in placements], [3, 8])
        actual = deepcopy(env)
        for b in placements:
            put_bomb(actual, b.row, b.col, b.owner, b.timer, b.blast_range)
        for tick in range(len(burns)):
            actual._tick_bombs_and_explode()
            expected = np.zeros_like(burns[tick])
            for pos in actual.flames:
                expected[pos] = True
            np.testing.assert_array_equal(burns[tick], expected)
            for pos in list(actual.flames):
                actual.flames[pos] -= 1
                if not actual.flames[pos]:
                    del actual.flames[pos]

    def test_counter_bomb_order_matches_real_wall_opening_when_AC_is_A1(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 1, 3
        env.agents[1].blast_range = 4
        env.grid[3, 3] = WOODEN_WALL
        with patch('rule_based_bot._survival_map', wraps=_survival_map) as predict:
            _can_bomb_and_escape(env, 1, 1, 3, None)
        placements = predict.call_args.args[3]
        burns, _ = env._forecast_blasts(placements)
        actual = deepcopy(env)
        for a in actual.agents:
            actual._try_place_bomb(a)
        for _ in range(8):
            actual._tick_bombs_and_explode()
        self.assertIn((4, 3), actual.flames, 'A0 opens the wall before A1 explodes')
        self.assertTrue(burns[7, 4, 3], 'The forecast must include the newly opened ray')

    def test_seeded_evaluation_reuses_boards_rotates_corners_and_restores_rng(self):
        np.random.seed(123)
        torch.manual_seed(123)
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        numpy_state, torch_state = np.random.get_state(), torch.get_rng_state()
        boards = []
        def policy(env, obs, aid):
            np.random.random()
            torch.rand(1)
            return ACTION_STOP
        def match(env, fns, **kwargs):
            obs = env.reset()
            aid = fns.index(policy)
            fns[aid](env, obs, aid)
            boards.append((aid, env.grid.copy()))
            wins = [int(i == aid) for i in range(2)]
            return dict(wins=wins, active_wins=wins, suicide_rates=[0, 0], draws=0,
                        avg_kills=wins, avg_bombs=[1, 1], avg_items=[1, 1], avg_steps=5)
        with patch('evaluate.run_match', side_effect=match):
            result = evaluate_seeded(policy, num_episodes=4, seed=8000)
        self.assertEqual([g['aid'] for g in result['games']], [0, 1, 0, 1])
        self.assertEqual([g['seed'] for g in result['games']], [8000, 8000, 8001, 8001])
        np.testing.assert_array_equal(boards[0][1], boards[1][1])
        self.assertEqual(result['active_win_rate'], 100)
        self.assertEqual(result['by_corner']['1']['num_episodes'], 2)
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        if cuda_state is not None:
            self.assertTrue(torch.equal(cuda_state, torch.cuda.get_rng_state()))
        with self.assertRaises(ValueError):
            evaluate_seeded(policy, num_episodes=3)

    def test_cancelled_validation_does_not_return_a_partial_score(self):
        stop = threading.Event()
        def cancel(done, total):
            stop.set()
        with patch('evaluate.run_match') as match:
            result = evaluate_seeded(lambda *args: ACTION_STOP, num_episodes=2,
                                     stop_event=stop, on_progress=cancel)
        self.assertIsNone(result)
        match.assert_not_called()

    def test_C_and_D_rotate_the_learner(self):
        for phase in ('C', 'D'):
            seen, metrics = [], []
            def observe(env, ep, aid, actions, opponents):
                if env.step_count == 0:
                    seen.append(aid)
            with patch('train_actor_critic.torch.save'), patch('train_actor_critic.os.makedirs'), patch('builtins.print'):
                train(phase=phase, episodes=2, board_size=7, max_steps=1, device='cpu',
                      checkpoint_dir='unused-test-checkpoints', run_report=False,
                      on_step=observe, on_episode=metrics.append)
            self.assertEqual(seen, [0, 1])
            self.assertEqual([m['learning_aid'] for m in metrics], [0, 1])

    def test_mask_preserves_exit_before_the_last_safe_move(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 1, 3
        env.grid[2, 3] = WOODEN_WALL
        put_bomb(env, 1, 5, 0, timer=3, blast_range=4)
        mask = get_action_mask(env, 0)
        self.assertFalse(mask[ACTION_STOP], 'Three moves needed; waiting loses the only exit')
        self.assertTrue(mask[ACTION_LEFT], 'An exploding transit cell can be crossed in time')
        self.assertEqual(_bfs_safe_cell(env, 1, 3, _build_danger_map(env)), ACTION_LEFT)
        for action in (ACTION_LEFT, ACTION_LEFT, ACTION_DOWN):
            env.step([action, ACTION_STOP])
        self.assertTrue(env.agents[0].alive)

    def test_exit_does_not_rely_on_winning_a_shared_destination(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 1, 3
        env.agents[1].row, env.agents[1].col = 1, 5
        put_bomb(env, 1, 1, 0, timer=4)
        mask = get_action_mask(env, 0)
        self.assertFalse(mask[ACTION_RIGHT], 'Opponent can also step into (1,4)')
        self.assertTrue(mask[ACTION_DOWN], 'Leave by an uncontested refuge')

    def test_shared_exit_is_attempted_early_instead_of_on_the_explosion_tick(self):
        from rule_based_bot import _safe_action_mask
        for learner, contest in [(0, False), (1, False), (0, True), (1, True)]:
            with self.subTest(learner=learner, contest=contest):
                env = arena(2)
                enemy = 1 - learner
                env.grid[1:6, 1:6] = RIGID_WALL
                for pos in [(1, 1), (2, 1), (3, 1), (1, 2), (1, 3), (1, 4), (1, 5)]:
                    env.grid[pos] = 0
                env.agents[0].row, env.agents[0].col = 1, 1
                env.agents[1].row, env.agents[1].col = 1, 3
                put_bomb(env, 3, 1, 0, timer=3)
                put_bomb(env, 1, 5, 1, timer=3)
                step = ACTION_RIGHT if learner == 0 else ACTION_LEFT
                mask = get_action_mask(env, learner)
                self.assertTrue(mask[step])
                self.assertFalse(mask[ACTION_STOP])
                self.assertFalse(mask[ACTION_BOMB])
                self.assertEqual(_bfs_safe_cell(env, 1, env.agents[learner].col,
                                               _build_danger_map(env), action_mask=mask), step)
                self.assertEqual(_bfs_safe_cell(env, 1, env.agents[learner].col,
                                               _build_danger_map(env)), step)
                baseline = _safe_action_mask(env, enemy, robust=False)
                self.assertTrue(baseline[ACTION_STOP])
                self.assertFalse(baseline[ACTION_LEFT])
                self.assertFalse(baseline[ACTION_RIGHT])
                bots = [RuleBasedAgent(0), RuleBasedAgent(1)]
                for tick in range(3):
                    actions = [ACTION_STOP, ACTION_STOP]
                    actions[enemy] = bots[enemy].select_action(env)
                    if contest and tick == 0:
                        actions[enemy] = ACTION_LEFT if learner == 0 else ACTION_RIGHT
                    if env.agents[learner].col != 2:
                        self.assertTrue(get_action_mask(env, learner)[step])
                        actions[learner] = step
                    env.step(actions)
                self.assertTrue(env.agents[learner].alive)
                self.assertFalse(env.agents[enemy].alive)

    def test_early_escape_waits_for_flames_but_keeps_urgent_escape(self):
        for timer, flame in [(4, True), (1, False)]:
            with self.subTest(timer=timer, flame=flame):
                env = arena(2)
                env.grid[1:6, 1:6] = RIGID_WALL
                for pos in [(1, 1), (2, 1), (3, 1), (1, 2), (1, 3)]:
                    env.grid[pos] = 0
                env.agents[0].row, env.agents[0].col = 1, 1
                env.agents[1].row, env.agents[1].col = 1, 3
                put_bomb(env, 3, 1, 0, timer=timer)
                if flame:
                    env.flames[(1, 2)] = 1
                mask = get_action_mask(env, 0)
                if flame:
                    self.assertFalse(mask[ACTION_RIGHT])
                    self.assertTrue(mask[ACTION_STOP])
                    self.assertEqual(_bfs_safe_cell(env, 1, 1, _build_danger_map(env), action_mask=mask), ACTION_STOP)
                else:
                    self.assertTrue(mask[ACTION_RIGHT])
                    self.assertFalse(mask[ACTION_STOP])

    def test_new_bomb_cannot_spend_the_only_tick_needed_to_escape_a_chain(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 3
        put_bomb(env, 3, 1, 1, timer=2)
        self.assertFalse(get_action_mask(env, 0)[ACTION_BOMB])
        put_bomb(env, 3, 3, 0, timer=8)
        self.assertEqual(_build_danger_map(env)[4, 3], 2, 'Chain shortens the nominal eight-tick fuse')

    def test_bomb_requires_a_wall_or_current_or_next_tick_opponent_target(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 5, 5
        self.assertFalse(get_action_mask(env, 0)[ACTION_BOMB], 'Empty-floor bombs provide no combat benefit')
        env.agents[1].row, env.agents[1].col = 3, 4
        self.assertTrue(get_action_mask(env, 0)[ACTION_BOMB], 'Preemptive placement can cover the next move')
        env.agents[1].row, env.agents[1].col = 5, 5
        env.grid[3, 3] = WOODEN_WALL
        self.assertTrue(get_action_mask(env, 0)[ACTION_BOMB], 'Keep useful route-opening bombs')

    def test_forecast_matches_real_chains_walls_and_flame_lifetimes(self):
        for seed in range(10):
            env = BombermanFFAEnv(board_size=9, num_agents=2, seed=seed, powerup_prob=0)
            env.reset()
            rng = np.random.default_rng(seed)
            empty = np.argwhere(env.grid == 0)
            for idx in rng.choice(len(empty), size=4, replace=False):
                r, c = empty[idx]
                put_bomb(env, int(r), int(c), 0, timer=int(rng.integers(1, 9)),
                         blast_range=int(rng.integers(1, 5)))
            env.flames[(1, 2)] = 1
            burns, blocked = env._forecast_blasts()
            actual = deepcopy(env)
            for tick in range(len(burns)):
                expected_blocked = np.array([[not actual._is_walkable(r, c) for c in range(9)]
                                             for r in range(9)])
                np.testing.assert_array_equal(blocked[tick], expected_blocked)
                actual._tick_bombs_and_explode()
                expected_burns = np.zeros((9, 9), dtype=bool)
                for pos in actual.flames:
                    expected_burns[pos] = True
                np.testing.assert_array_equal(burns[tick], expected_burns)
                for pos in list(actual.flames):
                    actual.flames[pos] -= 1
                    if actual.flames[pos] <= 0:
                        del actual.flames[pos]

    def test_useful_pickups_reward_once_and_stop_at_the_cap(self):
        for kind, attr, cap in [(POWERUP_AMMO, 'ammo', MAX_AMMO),
                                (POWERUP_RANGE, 'blast_range', MAX_BLAST_RANGE)]:
            env = arena(2)
            initial = getattr(env.agents[0], attr)
            total_bonus = 0
            for i in range(10):
                env.powerups[(1, 1)] = kind
                _, rewards, _, info = env.step([ACTION_STOP] * 2)
                upgraded = i < cap - initial
                self.assertEqual(info['items_collected'], {0: [kind]} if upgraded else {})
                self.assertAlmostEqual(rewards[0], .049 if upgraded else -.001)
                total_bonus += rewards[0] + .001
            self.assertEqual(getattr(env.agents[0], attr), cap)
            self.assertAlmostEqual(total_bonus, .05 * (cap - initial))

    def test_dead_agent_cannot_collect_an_item_in_flames(self):
        env = arena()
        env.powerups[(1, 1)] = POWERUP_AMMO
        put_bomb(env, 1, 1, 0)
        _, _, _, info = env.step([ACTION_STOP] * 3)
        self.assertFalse(env.agents[0].alive)
        self.assertEqual(env.agents[0].ammo, 1)
        self.assertEqual(info['items_collected'], {})

    def test_observation_exposes_capacity_free_slots_range_and_blasts(self):
        env = arena(2)
        before = env._get_obs(0)
        env.agents[0].ammo = 3
        env.agents[0].bombs_placed = 1
        env.agents[0].blast_range = 4
        after = env._get_obs(0)
        self.assertEqual(after.shape, (OBS_CHANNELS, 7, 7))
        np.testing.assert_array_equal(before[:8], after[:8])
        self.assertAlmostEqual(float(after[8, 1, 1]), 2/3)
        self.assertEqual(after[9, 1, 1], 1)
        self.assertEqual(after[10, 1, 1], 1)
        put_bomb(env, 3, 1, 1, timer=2, blast_range=3)
        self.assertEqual(env._get_obs(0)[11, 3, 3], .5)

    def test_old_checkpoint_migration_preserves_logits_and_value(self):
        torch.manual_seed(9)
        old = ActorCriticNet(in_channels=8, board_size=7).eval()
        new = ActorCriticNet(board_size=7).eval()
        with patch('train_actor_critic.torch.load', return_value=old.state_dict()):
            smart_load_weights(new, 'unused.pt', 'cpu')
        obs = torch.from_numpy(arena(2)._get_obs(0)).unsqueeze(0)
        with torch.no_grad():
            old_logits, old_value = old(obs[:, :8])
            new_logits, new_value = new(obs)
        torch.testing.assert_close(old_logits, new_logits)
        torch.testing.assert_close(old_value, new_value)
        self.assertEqual(int(torch.count_nonzero(new.conv[0].weight[:, 8:])), 0)

    def test_final_checkpoint_keeps_later_updates_after_best_episode(self):
        env = arena(2)
        env.max_steps = 1
        step = env.step
        episode = 0
        def winning_step(actions):
            nonlocal episode
            episode += 1
            obs, rewards, dones, info = step(actions)
            aid = (episode - 1) % 2
            env.agents[aid].kills = int(episode <= 200)
            env.agents[1-aid].alive = False
            info.update(alive=[a.alive for a in env.agents], episode_done=True,
                        kills=[(aid, 1-aid)] if episode <= 200 else [])
            rewards[aid] = 10.999
            return obs, rewards, dones, info
        net = ActorCriticNet(board_size=7)
        saved = {}
        def save(state, path):
            saved[Path(path).name] = {k: v.clone() for k, v in state.items()}
        def after_episode(data):
            if data['episode'] == 200:
                with torch.no_grad():
                    net.critic_head.bias.add_(25)
        validations = [dict(active_win_rate=a, win_rate=a, suicide_rate=s,
                            draw_rate=0, avg_kills=a/100, games=[]) for a, s in [(10, 20), (10, 10), (0, 10)]]
        selected_episodes = []
        def observe_selection(data):
            after_episode(data)
            if data['episode'] >= 200:
                selected_episodes.append(data['best_episode'])
        with patch('train_actor_critic.BombermanFFAEnv', return_value=env), \
             patch.object(env, 'step', side_effect=winning_step), \
             patch('train_actor_critic.ActorCriticNet', return_value=net), \
             patch('train_actor_critic.torch.save', side_effect=save), \
             patch('train_actor_critic.SnapshotOpponent', wraps=SnapshotOpponent) as snapshots, \
             patch('train_actor_critic.os.makedirs'), patch('builtins.print'), \
             patch('builtins.open', mock_open()), \
             patch('evaluate.evaluate_seeded', side_effect=validations) as validate:
            train(episodes=201, board_size=7, max_steps=1, device='cpu', log_every=1000,
                  checkpoint_dir='unused-test-checkpoints', run_report=False, on_episode=observe_selection,
                  validation_every=200, validation_episodes=2)
        self.assertEqual(validate.call_count, 3)
        self.assertEqual(snapshots.call_count, 3)
        for call in snapshots.call_args_list:
            self.assertIsNot(call.args[0], net, 'Validation must not share the learner network')
            self.assertEqual(call.args[1], 'cpu')
            self.assertEqual(next(call.args[0].parameters()).device.type, 'cpu')
        self.assertEqual(selected_episodes, [200, 200], 'Prefer fewer suicides at equal validation ActiveWR')
        self.assertFalse(torch.equal(saved['best_phaseC.pt']['critic_head.bias'],
                                     saved['final_phaseC.pt']['critic_head.bias']))

    def test_rulebot_uses_real_range_and_blocked_rays(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 3, 4
        env.agents[0].blast_range = 3
        bot = RuleBasedAgent(0)
        self.assertEqual(bot.select_action(env), ACTION_BOMB)
        env.grid[3, 2] = RIGID_WALL
        self.assertNotEqual(bot.select_action(env), ACTION_BOMB)

    def test_rulebot_and_teacher_collect_safe_useful_items_only(self):
        env = arena(2)
        env.agents[1].row, env.agents[1].col = 5, 5
        env.powerups[(1, 2)] = POWERUP_AMMO
        bot = RuleBasedAgent(0)
        self.assertEqual(bot.select_action(env), ACTION_RIGHT)
        self.assertEqual(get_attack_action(env, 0, _build_danger_map(env), get_action_mask(env, 0)), ACTION_RIGHT)
        env.flames[(1, 2)] = 1
        self.assertNotEqual(bot.select_action(env), ACTION_RIGHT)
        env.flames.clear()
        env.agents[0].ammo = MAX_AMMO
        from rule_based_bot import _bfs_toward_item
        self.assertIsNone(_bfs_toward_item(env, 0, _build_danger_map(env)))

    def test_evaluation_rejects_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            tournament(ac_model_path="missing-combat-checkpoint.pt")

    def test_direct_and_lingering_flames_credit_owner_once(self):
        for delayed in (False, True):
            with self.subTest(delayed=delayed):
                env = arena()
                env.agents[1].row, env.agents[1].col = (4, 3) if delayed else (3, 3)
                put_bomb(env, 3, 2, 0, blast_range=1)
                _, _, _, info = env.step([ACTION_STOP] * 3)
                if delayed:
                    self.assertEqual(info["kills"], [])
                    _, _, _, info = env.step([ACTION_STOP, ACTION_UP, ACTION_STOP])
                self.assertEqual(info["kills"], [(0, 1)])
                self.assertEqual(env.agents[0].kills, 1)
                _, _, _, info = env.step([ACTION_STOP] * 3)
                self.assertEqual(info["kills"], [])
                self.assertEqual(env.agents[0].kills, 1)
                env.reset()
                self.assertEqual(env._flame_owners, {})

    def test_overlapping_blasts_do_not_double_count_victim(self):
        env = arena()
        for a, pos in zip(env.agents, [(1, 3), (3, 3), (5, 3)]):
            a.row, a.col = pos
        put_bomb(env, 3, 1, 0, blast_range=3)
        put_bomb(env, 3, 5, 2, blast_range=3)
        _, _, _, info = env.step([ACTION_STOP] * 3)
        self.assertEqual(info["kills"], [(0, 1)])
        self.assertEqual([a.kills for a in env.agents], [1, 0, 0])

    def test_chain_reaction_credits_exploding_bomb_owner(self):
        env = arena()
        for a, pos in zip(env.agents, [(1, 5), (3, 5), (5, 5)]):
            a.row, a.col = pos
        put_bomb(env, 3, 1, 0)
        put_bomb(env, 3, 3, 2, timer=8)
        _, _, _, info = env.step([ACTION_STOP] * 3)
        self.assertEqual(info["kills"], [(2, 1)])
        self.assertEqual([a.kills for a in env.agents], [0, 0, 1])

    def test_suicide_is_penalized_and_is_not_a_kill(self):
        env = arena()
        env.agents[0].row, env.agents[0].col = 3, 3
        put_bomb(env, 3, 3, 0)
        _, rewards, _, info = env.step([ACTION_STOP] * 3)
        self.assertEqual(info["kills"], [(0, 0)])
        self.assertEqual(env.agents[0].kills, 0)
        self.assertEqual(_combat_reward(rewards[0], 0, info, 0), -2.5)

    def test_passive_win_has_no_bonus_even_after_a_token_bomb(self):
        info = {"kills": [(1, 1)], "alive": [True, False], "episode_done": True,
                "bombs_placed": [0]}
        self.assertAlmostEqual(_combat_reward(9.999, 0, info, 0), -0.001)
        info["kills"] = [(0, 1)]
        self.assertAlmostEqual(_combat_reward(10.999, 0, info, 1), 13.999)

    def test_attack_target_requires_clear_blast_and_escape(self):
        env = arena()
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 3, 3
        danger = _build_danger_map(env)
        self.assertEqual(get_attack_action(env, 0, danger, get_action_mask(env, 0)), ACTION_BOMB)
        env.grid[3, 2] = WOODEN_WALL
        self.assertNotIn((3, 3), env._blast_cells(3, 1, 2))
        env.grid[3, 2] = RIGID_WALL
        self.assertNotIn((3, 3), env._blast_cells(3, 1, 2))
        env.grid[1:-1, 1:-1] = RIGID_WALL
        env.grid[3, 1] = env.grid[3, 3] = 0
        mask = get_action_mask(env, 0)
        self.assertFalse(mask[ACTION_BOMB])
        self.assertIsNone(get_attack_action(env, 0, danger, mask))

    def test_danger_targets_escape_instead_of_attack(self):
        env = arena()
        env.flames[(1, 1)] = 1
        self.assertIsNone(get_attack_action_v1(env, 0, _build_danger_map(env), get_action_mask(env, 0)))

    def test_evaluation_separates_owned_kills_and_passive_wins(self):
        for owner in (0, 1):
            with self.subTest(owner=owner):
                env = arena(2)
                env.agents[0].row, env.agents[0].col = 1, 3
                env.agents[1].row, env.agents[1].col = 3, 1
                put_bomb(env, 3, 1, owner)
                obs = [env._get_obs(i) for i in range(2)]
                # Invalid BOMB attempts with exhausted ammo must not count as placements.
                fns = [lambda e, o, i: ACTION_BOMB if i == owner else ACTION_STOP] * 2
                with patch.object(env, "reset", return_value=obs):
                    stats = run_match(env, fns, num_episodes=1)
                self.assertEqual(stats["avg_bombs"], [0, 0])
                self.assertEqual(stats["win_rates"][0], 100)
                self.assertEqual(stats["active_win_rates"][0], 100 if owner == 0 else 0)
                self.assertEqual(stats["passive_wins"][0], int(owner == 1))
                self.assertEqual(stats["avg_kills"][0], int(owner == 0))
                self.assertEqual(stats["suicide_rates"][1], 100 if owner == 1 else 0)

    def test_evaluation_counts_only_successful_item_upgrades(self):
        env = arena(2)
        env.max_steps = 1
        env.powerups[(1, 1)] = POWERUP_AMMO
        obs = [env._get_obs(i) for i in range(2)]
        with patch.object(env, 'reset', return_value=obs):
            stats = run_match(env, [lambda e, o, i: ACTION_STOP] * 2, num_episodes=1)
        self.assertEqual(stats['avg_items'], [1, 0])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
