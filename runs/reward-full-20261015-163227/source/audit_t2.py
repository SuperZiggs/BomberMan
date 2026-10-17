"""Read-only T2 audit: fixed CNN traces, tied plans, clone checks and replay cases."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import shutil
from time import perf_counter

import numpy as np
import torch

from bomberman_env import BombermanFFAEnv, ACTION_STOP, ACTION_BOMB, _MOVE_DELTA
from evaluate import make_rule_based_fn
from rule_based_bot import _safe_action_mask, _bfs_safe_cell, _reachable_cells
import tactical_teacher
from train_actor_critic import ActorCriticNet, SnapshotOpponent, smart_load_weights, obs_to_tensor, get_action_mask
from visualize import capture_state

ROOT = Path(__file__).resolve().parent
T1 = ROOT / 'runs/two-bomb-complete-20261014-031626'
ACTION_NAMES = ('STOP', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def instrument_teacher():
    # Audit-only AST hook captures evaluated paths without editing the production teacher.
    tree = ast.parse(inspect.getsource(tactical_teacher.get_two_bomb_plan))
    fn = tree.body[0]
    fn.args.kwonlyargs.append(ast.arg(arg='audit_record'))
    fn.args.kw_defaults.append(ast.Constant(None))
    hook = ast.parse("""if audit_record is not None:
    audit_record(score, {'bombs': schedule, 'path': path, 'pressure': pressure,
                        'trapped': trapped, 'before': before, 'after': after, 'horizon': ticks})
""").body[0]
    inserted = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            for index, statement in enumerate(node.body):
                if (isinstance(statement, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'score' for t in statement.targets)):
                    node.body.insert(index + 1, hook)
                    inserted += 1
                    break
    if inserted != 1:
        raise RuntimeError('Teacher changed: audit hook must be reviewed before running')
    ast.fix_missing_locations(tree)
    scope = vars(tactical_teacher).copy()
    exec(compile(tree, '<T2 teacher instrumentation>', 'exec'), scope)
    return scope['get_two_bomb_plan'], ast.unparse(tree)


AUDIT_TEACHER, INSTRUMENTED_SOURCE = instrument_teacher()


def observed_teacher(env, aid, mask):
    score, tied = None, []
    def record(candidate_score, plan):
        nonlocal score, tied
        if score is None or candidate_score > score:
            score, tied = candidate_score, []
        if candidate_score == score:
            tied.append(plan)
    plan, stats = AUDIT_TEACHER(env, aid, mask, audit_record=record)
    assert stats['plans'] <= 64
    if plan:
        assert sorted({p['path'][0] for p in tied}) == plan['first_actions']
        assert all(mask[a] for a in plan['first_actions'])
    else:
        assert not tied
    return plan, stats, tied


def serial_plan(plan, start_tick):
    return {'path': list(plan['path']), 'bombs': [
        {'delay': delay, 'decision_tick': start_tick + delay, **asdict(bomb)}
        for delay, bomb in plan['bombs']],
        **{k: plan[k] for k in ('pressure', 'trapped', 'before', 'after', 'horizon')}}


def escape_suffix(env, aid, ticks):
    burns, blocked = env._forecast_blasts(min_horizon=max(1, ticks))
    blocked = tactical_teacher._learner_blockers(env, aid, blocked)
    start = (env.agents[aid].row, env.agents[aid].col)
    frontier, parents = {start}, []
    for tick in range(ticks):
        next_positions = {}
        for pos in sorted(frontier):
            for action, (dr, dc) in ((ACTION_STOP, (0, 0)), *_MOVE_DELTA.items()):
                dest = pos[0] + dr, pos[1] + dc
                if (0 <= dest[0] < env.board_size and 0 <= dest[1] < env.board_size
                        and not burns[tick][dest]
                        and (action == ACTION_STOP or not blocked[tick][dest])):
                    next_positions.setdefault(dest, (pos, action))
        if not next_positions:
            return None
        parents.append(next_positions)
        frontier = next_positions.keys()
    target, actions = min(frontier), []
    for parent in reversed(parents):
        target, action = parent[target]
        actions.append(action)
    return list(reversed(actions))


def opponent_action(env, aid, kind, bot):
    if not env.agents[aid].alive or kind == 'static':
        return ACTION_STOP
    if kind == 'rule':
        return bot.select_action(env)
    mask = _safe_action_mask(env, aid, robust=False)
    agent = env.agents[aid]
    danger = env._danger_map()
    if danger[agent.row, agent.col] > 0:
        action = _bfs_safe_cell(env, agent.row, agent.col, danger, robust=False)
        if action is not None and action != ACTION_BOMB and mask[action]:
            return action
    choices = [a for a in range(ACTION_BOMB) if mask[a]]
    return int(np.random.choice(choices)) if choices else ACTION_STOP


def reachable_comparison(env, aid, plan):
    schedules = {'baseline': (), 'combo': plan['bombs']}
    if len(plan['bombs']) == 2:
        schedules.update(first_only=plan['bombs'][:1], second_only=plan['bombs'][1:])
    forecasts, result = {}, {}
    for name, schedule in schedules.items():
        try:
            forecasts[name] = env._forecast_blasts(scheduled_bombs=schedule, min_horizon=plan['horizon'])
        except ValueError as exc:
            result[name] = {'error': str(exc)}
    horizon = max(len(burns) for burns, _ in forecasts.values())
    enemies = [a for a in env.agents if a.alive and a.agent_id != aid]
    for name in forecasts:
        burns, blocked = env._forecast_blasts(scheduled_bombs=schedules[name], min_horizon=horizon)
        result[name] = [int(np.count_nonzero(_reachable_cells((a.row, a.col), burns, blocked, horizon)))
                        for a in enemies]
    return {'common_horizon': horizon, 'survivors': result}


def check_plan(env, aid, plan, kind, enforce_mask):
    from rule_based_bot import RuleBasedAgent
    numpy_state = np.random.get_state()
    started = perf_counter()
    try:
        with torch.random.fork_rng(devices=[]):
            branch, bot = deepcopy(env), RuleBasedAgent(1 - aid)
            prefix = plan['path']
            placements = {delay: b for delay, b in plan['bombs']}
            result = {'opponent': kind, 'enforce_mask': enforce_mask, 'prefix_status': 'complete',
                      'first_bomb': False, 'second_bomb': False, 'blocked_tick': None,
                      'blocked_action': None, 'mask': None, 'suffix_found': None,
                      'full_horizon_alive': None, 'terminal_tick': None, 'post_terminal_physics': False,
                      'new_enemy_bombs': 0, 'range_changed': False, 'action_tape': [], 'suffix': None}
            for tick, action in enumerate(prefix):
                if branch._episode_done:
                    result.update(prefix_status='terminal_before_prefix', terminal_tick=tick)
                    break
                if enforce_mask:
                    mask = get_action_mask(branch, aid)
                    if not mask[action]:
                        result.update(prefix_status='mask_blocked', blocked_tick=tick,
                                      blocked_action=action, mask=list(map(bool, mask)))
                        break
                before = branch.agents[aid].row, branch.agents[aid].col
                actions = [0, 0]
                actions[aid], actions[1 - aid] = action, opponent_action(branch, 1 - aid, kind, bot)
                result['action_tape'].append(actions)
                _, _, _, info = branch.step(actions)
                result['new_enemy_bombs'] += int(1 - aid in info['bombs_placed'])
                if action == ACTION_BOMB:
                    expected = placements[tick]
                    placed = aid in info['bombs_placed'] and before == (expected.row, expected.col)
                    if not placed:
                        result.update(prefix_status='placement_failed', blocked_tick=tick)
                        break
                    number = sorted(placements).index(tick)
                    result['first_bomb' if number == 0 else 'second_bomb'] = True
                    result['range_changed'] |= branch.agents[aid].blast_range != expected.blast_range
                elif action in _MOVE_DELTA:
                    dr, dc = _MOVE_DELTA[action]
                    if (branch.agents[aid].row, branch.agents[aid].col) != (before[0] + dr, before[1] + dc):
                        result.update(prefix_status='movement_failed', blocked_tick=tick)
                        break
                if not branch.agents[aid].alive:
                    result.update(prefix_status='learner_died', blocked_tick=tick, full_horizon_alive=False)
                    break
                if info['episode_done']:
                    result['terminal_tick'] = tick + 1
            if result['prefix_status'] == 'complete':
                remaining = max(0, plan['horizon'] - len(prefix))
                suffix = escape_suffix(branch, aid, remaining)
                result['suffix'] = suffix
                result['suffix_found'] = suffix is not None
                if suffix is not None:
                    for action in suffix:
                        # Only continue physics after terminal once the real-game prefix completed.
                        result['post_terminal_physics'] |= branch._episode_done
                        actions = [0, 0]
                        actions[aid], actions[1 - aid] = action, opponent_action(branch, 1 - aid, kind, bot)
                        result['action_tape'].append(actions)
                        _, _, _, info = branch.step(actions)
                        result['new_enemy_bombs'] += int(1 - aid in info['bombs_placed'])
                        if not branch.agents[aid].alive:
                            break
                    result['full_horizon_alive'] = branch.agents[aid].alive
            if not branch.agents[aid].alive:
                result['full_horizon_alive'] = False
            result.update(learner_alive=branch.agents[aid].alive,
                          final_positions=[[a.row, a.col] for a in branch.agents],
                          owned_kills_in_branch=branch.agents[aid].kills - env.agents[aid].kills,
                          elapsed_ms=(perf_counter() - started) * 1000)
            return result
    finally:
        np.random.set_state(numpy_state)


class PlanTracker:
    def __init__(self, plan, tick, plan_id, env, aid):
        self.plan, self.start, self.plan_id = plan, tick, plan_id
        self.aid, self.start_kills = aid, env.agents[aid].kills
        self.row = {'plan_id': plan_id, 'start_tick': tick, 'two_bomb': len(plan['bombs']) == 2,
                    'first_bomb': False, 'second_bomb': False, 'strict_prefix': True,
                    'prefix_deviations': 0, 'movement_failures': 0,
                    'failure_reason': None, 'status': None, 'churn_transitions': 0,
                    'observed_transitions': 0, 'churn_reasons': Counter(), 'trapped': plan['trapped']}

    def compare(self, tick, tied, events):
        remaining = {(self.start + d, b.row, b.col) for d, b in self.plan['bombs'] if self.start + d >= tick}
        if not remaining:
            return
        self.row['observed_transitions'] += 1
        matches = any(remaining.issubset({(tick + d, b.row, b.col) for d, b in p['bombs']}) for p in tied)
        if not matches:
            self.row['churn_transitions'] += 1
            reason = '+'.join(events) if events else 'unexplained_by_logged_events'
            self.row['churn_reasons'][reason] += 1

    def advance(self, tick, action, before, env, info):
        offset, aid = tick - self.start, self.aid
        if offset < len(self.plan['path']) and action != self.plan['path'][offset]:
            self.row['strict_prefix'] = False
            self.row['prefix_deviations'] += 1
        if offset < len(self.plan['path']) and action in _MOVE_DELTA:
            dr, dc = _MOVE_DELTA[action]
            if (env.agents[aid].row, env.agents[aid].col) != (before[0] + dr, before[1] + dc):
                self.row['strict_prefix'] = False
                self.row['movement_failures'] += 1
        if offset in {d for d, _ in self.plan['bombs']}:
            number = [d for d, _ in self.plan['bombs']].index(offset)
            bomb = self.plan['bombs'][number][1]
            placed = action == ACTION_BOMB and aid in info['bombs_placed'] and before == (bomb.row, bomb.col)
            self.row['first_bomb' if number == 0 else 'second_bomb'] = placed
            if not placed:
                self.row['failure_reason'] = self.row['failure_reason'] or 'scheduled_placement_not_completed'
        if not env.agents[aid].alive:
            self.row['failure_reason'] = self.row['failure_reason'] or 'learner_died'
        due = offset + 1 >= self.plan['horizon']
        if not due and not info['episode_done']:
            return None
        completed = self.row['first_bomb'] and (not self.row['two_bomb'] or self.row['second_bomb'])
        last_bomb = self.plan['bombs'][-1][0]
        if self.row['failure_reason']:
            status = 'failure'
        elif completed:
            status = 'success'
        elif info['episode_done'] and offset < last_bomb:
            status = 'early_terminal'
        else:
            status = 'failure'
        self.row.update(status=status, end_tick=tick + 1, learner_alive=env.agents[aid].alive,
                        full_horizon_observed=due, end_reason='terminal' if info['episode_done'] else 'horizon',
                        terminal_win=int(env.agents[aid].alive and not env.agents[1 - aid].alive),
                        terminal_draw=int(sum(info['alive']) != 1),
                        owned_kills_in_window=env.agents[aid].kills - self.start_kills,
                        terminal=info['episode_done'], complete_placements=completed,
                        churn_reasons=dict(self.row['churn_reasons']))
        return self.row


def load_net(path):
    net = ActorCriticNet(board_size=9)
    smart_load_weights(net, str(path), 'cpu')
    net.eval()
    return net


@torch.no_grad()
def replay_game(net, seed, aid, max_steps, traced=True, check_every=10, max_checks=50):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=max_steps, seed=seed)
    obs = env.reset()
    bot = make_rule_based_fn(1 - aid)
    policy = SnapshotOpponent(net, 'cpu')
    trace, trackers, checks, tape = [], [], [], []
    tracker, sampled, suicide, bomb_count, item_count = None, 0, False, 0, 0
    labels = ['CNN' if i == aid else 'RuleBot' for i in range(2)]
    last_events = []
    for tick in range(max_steps):
        actions, logits_list, probabilities, mask = [0, 0], None, None, None
        for index in range(2):
            if index != aid:
                actions[index] = bot(env, obs, index)
            elif traced:
                logits, _ = net(obs_to_tensor(obs[aid], 'cpu'))
                mask = get_action_mask(env, aid)
                dist = torch.distributions.Categorical(logits=logits.masked_fill(~torch.tensor(mask, dtype=torch.bool), -1e9))
                actions[aid] = dist.sample().item()
                logits_list, probabilities = logits[0].tolist(), dist.probs[0].tolist()
            else:
                actions[aid] = policy.select_action(obs[aid], env, aid)
        before_pos = env.agents[aid].row, env.agents[aid].col
        enemy_pos = env.agents[1 - aid].row, env.agents[1 - aid].col
        before_frame = capture_state(env, labels, actions) if traced else None
        step_checks, tracker_id = [], None
        if traced:
            fingerprint = hashlib.sha256(pickle.dumps(env.__dict__)).hexdigest()
            plan, stats, tied = observed_teacher(env, aid, mask)
            if tracker:
                tracker.compare(tick, tied, last_events)
            if tracker is None and plan and actions[aid] in plan['first_actions']:
                selected = next(p for p in tied if p['path'][0] == actions[aid])
                tracker = PlanTracker(selected, tick, f'{seed}-A{aid}-t{tick}', env, aid)
            if tracker:
                tracker_id = tracker.plan_id
            if plan and tick % check_every == 0 and sampled < max_checks:
                selected = next((p for p in tied if p['path'][0] == actions[aid]), tied[0])
                quality = reachable_comparison(env, aid, selected)
                for kind in ('static', 'evader', 'rule'):
                    for enforce in (False, True):
                        check = check_plan(env, aid, selected, kind, enforce)
                        check.update(seed=seed, aid=aid, tick=tick, state_id=fingerprint,
                                     plan=serial_plan(selected, tick), reachability=quality)
                        step_checks.append(check)
                sampled += 1
            assert hashlib.sha256(pickle.dumps(env.__dict__)).hexdigest() == fingerprint
            checks.extend(step_checks)
        prefix_deviation = (tracker is not None and tick - tracker.start < len(tracker.plan['path'])
                            and actions[aid] != tracker.plan['path'][tick - tracker.start])
        obs, rewards, _, info = env.step(actions)
        tape.append(actions)
        bomb_count += int(aid in info['bombs_placed'])
        item_count += len(info['items_collected'].get(aid, []))
        suicide |= aid in info.get('self_blast_deaths', []) or (aid, aid) in info['kills']
        after_frame = capture_state(env, labels, actions, rewards) if traced else None
        completed = tracker.advance(tick, actions[aid], before_pos, env, info) if tracker else None
        if completed:
            trackers.append(completed)
            tracker = None
        if traced:
            trace.append({'tick': tick, 'state_id': fingerprint, 'actions': actions,
                          'logits': logits_list, 'probabilities': probabilities, 'mask': list(map(bool, mask)),
                          'first_actions': plan['first_actions'] if plan else [], 'stats': stats,
                          'tied_plans': [serial_plan(p, tick) for p in tied],
                          'before': before_frame, 'after': after_frame, 'info': info,
                          'active_plan_id': tracker_id, 'completed_plan': completed, 'checks': step_checks})
        last_events = []
        if prefix_deviation:
            last_events.append('learner_prefix_deviation')
        if traced and (before_frame['flames'] != after_frame['flames']
                       or len(before_frame['bombs']) > len(after_frame['bombs'])):
            last_events.append('known_hazard_changed')
        if enemy_pos != (env.agents[1 - aid].row, env.agents[1 - aid].col):
            last_events.append('enemy_moved')
        if 1 - aid in info['bombs_placed']:
            last_events.append('enemy_bomb')
        if info['walls_destroyed']:
            last_events.append('wood_changed')
        if info['items_collected']:
            last_events.append('item_collected')
        if actions[aid] in _MOVE_DELTA:
            dr, dc = _MOVE_DELTA[actions[aid]]
            if (env.agents[aid].row, env.agents[aid].col) != (before_pos[0] + dr, before_pos[1] + dc):
                last_events.append('learner_movement_failed')
        if info['episode_done']:
            break
    if tracker:
        tracker.row.update(status='censored', churn_reasons=dict(tracker.row['churn_reasons']))
        trackers.append(tracker.row)
    win = int(env.agents[aid].alive and not env.agents[1 - aid].alive)
    result = {'seed': seed, 'aid': aid, 'win': win, 'active_win': int(win and env.agents[aid].kills > 0),
              'suicide': int(suicide), 'draw': int(sum(info['alive']) != 1), 'kills': env.agents[aid].kills,
              'steps': info['step'], 'bombs': bomb_count, 'items': item_count}
    return result, tape, trace, trackers, checks


def rate(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'percent': 100 * numerator / denominator if denominator else None}


def distribution(values):
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'p95': None, 'max': None}
    return {'count': len(values), 'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'max': float(max(values))}


def diagnostic_row(row):
    labels = row['first_actions']
    plan = row['tied_plans'][0] if labels else None
    aid = row['aid']
    return {'aid': aid, 'labels': labels, 'agreement': row['actions'][aid] in labels,
            'mass': sum(row['probabilities'][a] for a in labels),
            'uniform_chance': len(labels) / sum(row['mask']), 'mask_support': sum(row['mask']),
            'two_bomb': len(plan['bombs']) == 2 if plan else False,
            'pressure': plan['pressure'] if plan else None, 'trapped': bool(plan and plan['trapped']),
            'stats': row['stats']}


def summarize(games, diagnostics, trackers, checks):
    accepted = [d for d in diagnostics if d['labels']]
    two = [p for p in trackers if p['two_bomb']]
    two_first = [p for p in two if p['first_bomb']]
    outcomes = {key: rate(sum(g[key] for g in games), len(games))
                for key in ('win', 'active_win', 'draw', 'suicide')}
    outcomes['passive_win'] = rate(sum(g['win'] - g['active_win'] for g in games), len(games))
    outcomes.update({f'avg_{key}': distribution([g[key] for g in games])['mean']
                     for key in ('kills', 'bombs', 'items', 'steps')})
    rejected = Counter()
    for d in diagnostics:
        rejected.update(d['stats']['rejected'])
    clones = {}
    for kind in ('static', 'evader', 'rule'):
        for enforced in (False, True):
            group = [c for c in checks if c['opponent'] == kind and c['enforce_mask'] == enforced]
            known = [c for c in group if c['full_horizon_alive'] is not None]
            clones[f'{kind}_{"masked" if enforced else "physical"}'] = {
                'checks': len(group), 'prefix_status': dict(Counter(c['prefix_status'] for c in group)),
                'prefix_pass': rate(sum(c['prefix_status'] == 'complete' for c in group), len(group)),
                'survival_observed': rate(sum(c['full_horizon_alive'] for c in known), len(known)),
                'survival_unknown': len(group) - len(known),
                'post_terminal_physics': sum(c['post_terminal_physics'] for c in group),
                'new_enemy_bombs': sum(c['new_enemy_bombs'] > 0 for c in group),
                'mask_blocked_actions': dict(Counter(ACTION_NAMES[c['blocked_action']] for c in group
                                                   if c['prefix_status'] == 'mask_blocked')),
                'elapsed_ms': distribution([c['elapsed_ms'] for c in group])}
    physical = {(c['seed'], c['aid'], c['tick'], c['opponent']): c
                for c in checks if not c['enforce_mask']}
    gaps = [c for c in checks if c['enforce_mask'] and c['prefix_status'] == 'mask_blocked'
            and physical[(c['seed'], c['aid'], c['tick'], c['opponent'])]['prefix_status'] == 'complete']
    reasons = Counter()
    for p in trackers:
        reasons.update(p['churn_reasons'])
    quality = [c['reachability']['survivors'] for c in checks
               if c['opponent'] == 'static' and not c['enforce_mask'] and len(c['plan']['bombs']) == 2]
    valid_quality = [q for q in quality if isinstance(q.get('first_only'), list)
                     and isinstance(q.get('second_only'), list)]
    synergy = sum(any(b > 0 and a == 0 and f > 0 and s > 0
                      for b, a, f, s in zip(q['baseline'], q['combo'], q['first_only'], q['second_only']))
                  for q in valid_quality)
    return {'games': len(games), 'outcomes': outcomes, 'queries': len(diagnostics),
            'coverage': rate(len(accepted), len(diagnostics)),
            'agreement': rate(sum(d['agreement'] for d in accepted), len(accepted)),
            'target_probability_mass': distribution([d['mass'] for d in accepted]),
            'uniform_random_agreement': distribution([d['uniform_chance'] for d in accepted]),
            'label_support': dict(Counter(len(d['labels']) for d in accepted)),
            'mask_support': dict(Counter(d['mask_support'] for d in accepted)),
            'label_action_participation': dict(Counter(ACTION_NAMES[a] for d in accepted for a in d['labels'])),
            'two_bomb_labels': rate(sum(d['two_bomb'] for d in accepted), len(accepted)),
            'trap_labels': rate(sum(d['trapped'] for d in accepted), len(accepted)),
            'trap_one_bomb': rate(sum(d['trapped'] for d in accepted if not d['two_bomb']),
                                  sum(not d['two_bomb'] for d in accepted)),
            'trap_two_bomb': rate(sum(d['trapped'] for d in accepted if d['two_bomb']),
                                  sum(d['two_bomb'] for d in accepted)),
            'pressure': distribution([d['pressure'] for d in accepted]),
            'pressure_below_0_5': rate(sum(d['pressure'] < 0.5 for d in accepted), len(accepted)),
            'plans_evaluated': sum(d['stats']['plans'] for d in diagnostics),
            'rejected_candidates': dict(rejected),
            'instrumented_teacher_ms': distribution([d['stats']['milliseconds'] for d in diagnostics]),
            'plan_starts': len(trackers), 'two_bomb_starts': len(two),
            'first_bomb': rate(sum(p['first_bomb'] for p in trackers), len(trackers)),
            'second_bomb_given_first': rate(sum(p['second_bomb'] for p in two_first), len(two_first)),
            'two_bomb_complete_placements': rate(sum(p.get('complete_placements', False) for p in two), len(two)),
            'strict_two_bomb_completion': rate(sum(p.get('complete_placements', False) and p['strict_prefix']
                                                    for p in two), len(two)),
            'plan_status': dict(Counter(p['status'] for p in trackers)),
            'windows_ended_before_horizon': sum(not p.get('full_horizon_observed', False) for p in trackers),
            'learner_death_windows': rate(sum(p.get('learner_alive') is False for p in trackers), len(trackers)),
            'plan_failure_reasons': dict(Counter(p['failure_reason'] for p in trackers if p['failure_reason'])),
            'owned_kill_windows': rate(sum(p.get('owned_kills_in_window', 0) > 0 for p in trackers), len(trackers)),
            'churn': rate(sum(p['churn_transitions'] for p in trackers),
                          sum(p['observed_transitions'] for p in trackers)),
            'churn_event_associations': dict(reasons), 'clone_checks': clones,
            'physical_pass_but_mask_blocked': len(gaps),
            'sampled_two_bomb_quality': len(quality),
            'two_bomb_trap_both_singles_leave_refuges': rate(synergy, len(valid_quality)),
            'single_bomb_counterfactual_invalid': len(quality) - len(valid_quality)}


def replay_categories(result, trace, trackers, checks):
    categories = []
    if any(p['two_bomb'] and p['first_bomb'] and not p['second_bomb'] for p in trackers):
        categories.append('first_bomb_without_second')
    physical = {(c['tick'], c['opponent']): c for c in checks if not c['enforce_mask']}
    if any(c['enforce_mask'] and c['prefix_status'] == 'mask_blocked'
           and physical[(c['tick'], c['opponent'])]['prefix_status'] == 'complete' for c in checks):
        categories.append('physical_pass_mask_blocked')
    if any(p['churn_reasons'].get('unexplained_by_logged_events', 0) for p in trackers):
        categories.append('churn_unknown')
    if result['draw'] and any(r['first_actions'] and not r['tied_plans'][0]['trapped'] for r in trace):
        categories.append('pressure_without_trap_draw')
    if any(c['opponent'] == 'rule' and c['prefix_status'] not in ('complete', 'terminal_before_prefix')
           and c['new_enemy_bombs'] for c in checks):
        categories.append('counter_bomb_prefix_failure')
    if any(p.get('complete_placements') and p.get('owned_kills_in_window', 0) > 0 for p in trackers):
        categories.append('placements_with_owned_kill')
    if result['suicide']:
        categories.append('learner_self_blast')
    return categories or ['coverage_example']


def select_replays(pool, count):
    chosen, seen = [], set()
    def add(case):
        if case['game_id'] not in seen and len(chosen) < count:
            chosen.append(case)
            seen.add(case['game_id'])
    # Include each model/corner, then the first examples of the registered categories.
    for key, case in pool.items():
        if key[0] == 'coverage_example':
            add(case)
    for case in pool.values():
        add(case)
    return chosen


def write_replays(output, cases):
    dump(output / 'replay_cases.json', cases)
    data = json.dumps(cases, ensure_ascii=False).replace('<', '\\u003c')
    page = """<!doctype html><html lang="vi"><meta charset="utf-8">
<title>T2 replay audit</title><style>
body{font:16px system-ui;background:#101827;color:#edf3ff;margin:24px}
button,select,input{font:inherit;margin:6px;padding:6px}canvas{background:#172338;max-width:100%}
.layout{display:flex;gap:24px;flex-wrap:wrap}pre{white-space:pre-wrap;max-width:680px}
</style><h1>T2 replay audit</h1><p>Cyan: đường CNN đã đi. Tím: prefix và vị trí bom dự báo.
Board hiển thị trước action; timeline ghi kết quả sau action. Kill trong window chưa chứng minh do combo.</p>
<label>Trận <select id="games"></select></label><button id="prev">←</button>
<button id="play">Chạy / dừng</button><button id="next">→</button>
<label>Tick <input id="tick" type="range" min="0" value="0"></label>
<div class="layout"><canvas id="board" width="450" height="450"></canvas><pre id="details"></pre></div>
<script>
const cases=__DATA__, names=['STOP','UP','DOWN','LEFT','RIGHT','BOMB'];
const gs=document.getElementById('games'), slider=document.getElementById('tick');
const canvas=document.getElementById('board'), ctx=canvas.getContext('2d');let timer=null;
cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;
o.textContent=c.game_id+' | '+c.categories.join(', ');gs.appendChild(o)});
function draw(){if(!cases.length)return;const c=cases[+gs.value],r=c.timeline[+slider.value],f=r.before;
const n=f.grid.length,s=450/n;ctx.clearRect(0,0,450,450);
f.grid.forEach((row,y)=>row.forEach((v,x)=>{ctx.fillStyle=v===1?'#667085':v===2?'#aa7044':'#172338';
ctx.fillRect(x*s,y*s,s-1,s-1)}));
function circle(row,col,color,label,radius=.3){ctx.beginPath();ctx.fillStyle=color;
ctx.arc((col+.5)*s,(row+.5)*s,s*radius,0,Math.PI*2);ctx.fill();ctx.fillStyle='#fff';
ctx.font='14px system-ui';ctx.textAlign='center';ctx.fillText(label,(col+.5)*s,(row+.6)*s)}
f.flames.forEach(p=>{ctx.fillStyle='#ff8f38';ctx.fillRect(p[1]*s,p[0]*s,s-1,s-1)});
f.powerups.forEach(p=>circle(p[0][0],p[0][1],'#477b62',String(p[1]),.2));
f.bombs.forEach(b=>circle(b.row,b.col,'#070b12',String(b.timer)));
ctx.strokeStyle='#55d9dc';ctx.lineWidth=2;ctx.beginPath();
c.timeline.slice(0,+slider.value+1).forEach((q,i)=>{const a=q.before.agents[c.result.aid];
if(i)ctx.lineTo((a.col+.5)*s,(a.row+.5)*s);else ctx.moveTo((a.col+.5)*s,(a.row+.5)*s)});ctx.stroke();
const plan=r.tied_plans.find(p=>p.path[0]===r.actions[c.result.aid])||r.tied_plans[0];
if(plan){let a=f.agents[c.result.aid],y=a.row,x=a.col;const ds=[[0,0],[-1,0],[1,0],[0,-1],[0,1],[0,0]];
ctx.strokeStyle='#d290ee';ctx.lineWidth=3;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo((x+.5)*s,(y+.5)*s);
plan.path.forEach(action=>{y+=ds[action][0];x+=ds[action][1];ctx.lineTo((x+.5)*s,(y+.5)*s)});
ctx.stroke();ctx.setLineDash([]);plan.bombs.forEach((b,i)=>circle(b.row,b.col,'#9b5ab9',String(i+1),.22))}
f.agents.forEach(a=>{if(a.alive)circle(a.row,a.col,a.agent_id===c.result.aid?'#2699b7':'#d15164','A'+a.agent_id)});
document.getElementById('details').textContent=JSON.stringify({game:c.game_id,result:c.result,tick:r.tick,
actions:r.actions.map(a=>names[a]),mask:r.mask,labels:r.first_actions.map(a=>names[a]),
probabilities:r.probabilities,plan,active_plan:r.active_plan_id,completed_plan:r.completed_plan,
events:r.info,after_agents:r.after.agents,clone_checks:r.checks},null,2)}
function changeGame(){slider.max=Math.max(0,cases[+gs.value].timeline.length-1);slider.value=0;draw()}
function advance(d){slider.value=Math.max(0,Math.min(+slider.max,+slider.value+d));draw()}
gs.onchange=changeGame;slider.oninput=draw;document.getElementById('prev').onclick=()=>advance(-1);
document.getElementById('next').onclick=()=>advance(1);document.getElementById('play').onclick=()=>{
if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{if(+slider.value===+slider.max){clearInterval(timer);timer=null}else advance(1)},200)};
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')advance(1);if(e.key==='ArrowLeft')advance(-1)});
if(cases.length)changeGame();
</script></html>"""
    (output / 'replays.html').write_text(page.replace('__DATA__', data), encoding='utf-8')


def write_report(output, summary):
    lines = ['# T2 fixed-policy audit', '',
             f'Audit thu đủ dữ liệu theo config; full_protocol={summary["full_protocol"]}. Quyết định sửa teacher chờ đọc trace/replay. Giữ checkpoint gốc.', '',
             '| Model | Trận | ActiveWR | Hòa | Tự nổ | Coverage | Agreement | Bom hai / bom đầu |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    def pct(value):
        return 'N/A' if value['percent'] is None else f"{value['percent']:.2f}% ({value['numerator']}/{value['denominator']})"
    for model, entry in summary['models'].items():
        s = entry['overall']
        lines.append(f"| {model} | {s['games']} | {pct(s['outcomes']['active_win'])} | {pct(s['outcomes']['draw'])} | "
                     f"{pct(s['outcomes']['suicide'])} | {pct(s['coverage'])} | {pct(s['agreement'])} | "
                     f"{pct(s['second_bomb_given_first'])} |")
    lines += ['', '## Cách đọc để chọn một nhánh', '',
              '- L1: xem pressure, trap và hiệu quả clone với evader trước khi thử lọc nhãn.',
              '- L2: xem physical_pass_but_mask_blocked, tick/action bị chặn; chưa nới mask.',
              '- L3: xem churn và replay. Event association chỉ là proxy; unknown chưa chứng minh teacher đổi vô lý.',
              '- T4W: nếu plan khả thi nhưng agreement/mass thấp, kiểm tra nhãn và cân bằng gradient trước.',
              '- L4: xem timing instrument/clone/write riêng; observer không tương đương timing train T1.', '',
              f"Replay: {summary['replay_cases']} games, xem replays.html. Mỗi model có breakdown A0/A1 trong summary.json.", '',
              '## Giới hạn', '', *['- ' + note for note in summary['limitations']]]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boards', type=int, default=100, help='board seeds per model; both A0/A1')
    parser.add_argument('--seed', type=int, default=28000)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--check-every', type=int, default=10)
    parser.add_argument('--max-checks', type=int, default=50, help='sampled states/game, six branches/state')
    parser.add_argument('--output', type=Path, help='new directory only; default timestamp under runs/')
    args = parser.parse_args(argv)
    if min(args.boards, args.max_steps, args.check_every, args.max_checks) < 1 or args.seed < 0:
        parser.error('counts must be positive and seed nonnegative')
    torch.set_num_threads(1)
    checkpoints = {'original': ROOT / 'runs/pressure-C-next/final_phaseC.pt',
                   'two_bomb_final': T1 / 'two_bomb/final_phaseC.pt',
                   'two_bomb_best': T1 / 'two_bomb/best_phaseC.pt'}
    paths = {**{name: ROOT / name for name in ('bomberman_env.py', 'rule_based_bot.py',
             'train_actor_critic.py', 'tactical_teacher.py', 'evaluate.py', 'visualize.py',
             'pilot_two_bomb.py', 'audit_t2.py', 'test_audit_t2.py')},
             **checkpoints}
    hashes = {name: digest(path) for name, path in paths.items()}
    t1_config = json.loads((T1 / 'config.json').read_text(encoding='utf-8'))
    for name, expected in t1_config['source_sha256'].items():
        if hashes[name] != expected:
            raise RuntimeError(f'{name} changed since T1; audit provenance must be reviewed')
    if hashes['original'] != t1_config['warm_start_sha256']:
        raise RuntimeError('Original checkpoint changed since T1')
    nets = {name: load_net(path) for name, path in checkpoints.items()}
    equal = all(torch.equal(value, nets['two_bomb_best'].state_dict()[key])
                for key, value in nets['original'].state_dict().items())
    if not equal:
        raise RuntimeError('Best no longer equals original; review T1 before using this two-model protocol')
    del nets['two_bomb_best']
    output = (args.output or ROOT / ('runs/t2-audit-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
    if Path(str(output) + '-send.zip').exists():
        raise FileExistsError('Send ZIP already exists; choose a new output directory')
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    config = {'started_utc': datetime.now(timezone.utc).isoformat(), 'output': str(output),
              'boards_per_model': args.boards, 'seed': args.seed, 'last_seed': args.seed + args.boards - 1,
              'corners': [0, 1], 'max_steps': args.max_steps, 'board_size': 9, 'num_agents': 2,
              'device': 'cpu', 'torch_threads': 1, 'torch': torch.__version__, 'numpy': np.__version__,
              'check_every': args.check_every, 'max_sampled_states_per_game': args.max_checks,
              'opponents': ['static', 'evader', 'rule'], 'sampling': 'plan exists AND tick modulo check_every is zero',
              'checkpoints': {name: str(path) for name, path in checkpoints.items()}, 'sha256': hashes,
              'aliases': {'two_bomb_best': 'original'}, 'best_equals_original_tensors': equal,
              'plan_choice': 'first tied path matching sampled first action; canonical order, no extra sampling',
              'tracker': 'one active window; exact placement ticks/positions; early terminals kept in denominators',
              'replay_selection': 'first each model/corner, first registered category examples, then earliest distinct games',
              'audit_set': 'fixed policies after training, not historical training trajectories',
              'full_protocol': args.boards == 100 and args.max_steps == 500 and args.check_every == 10 and args.max_checks == 50}
    dump(output / 'config.json', config)
    source_dir = output / 'source'
    source_dir.mkdir()
    for name, path in paths.items():
        if name.endswith('.py'):
            shutil.copyfile(path, source_dir / name)
    (source_dir / 'teacher_instrumented.py').write_text(INSTRUMENTED_SOURCE, encoding='utf-8')
    print(f'Output: {output}', flush=True)
    print(f'Games: {4 * args.boards}; six clone branches per sampled state. CPU one thread.', flush=True)
    dump(output / 'status.json', {'status': 'running', 'completed_games': 0})
    models, all_games, parity, pool = {}, {}, [], {}
    completed_games, write_seconds = 0, 0.0
    try:
        with gzip.open(output / 'trace.jsonl.gz', 'wt', encoding='utf-8') as trace_file, \
             gzip.open(output / 'plan_checks.jsonl.gz', 'wt', encoding='utf-8') as check_file, \
             (output / 'plan_starts.jsonl').open('w', encoding='utf-8') as plan_file, \
             (output / 'games.jsonl').open('w', encoding='utf-8') as game_file:
            for model, net in nets.items():
                games, diagnostics, trackers, checks = [], [], [], []
                for seed in range(args.seed, args.seed + args.boards):
                    for aid in (0, 1):
                        result, tape, trace, starts, branches = replay_game(net, seed, aid, args.max_steps,
                            check_every=args.check_every, max_checks=args.max_checks)
                        # Both corner tapes and the final RNG must match the unobserved policy.
                        if seed == args.seed:
                            np_before, torch_before = pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
                            plain_result, plain_tape, *_ = replay_game(net, seed, aid, args.max_steps, traced=False)
                            passed = (result == plain_result and tape == plain_tape
                                      and np_before == pickle.dumps(np.random.get_state())
                                      and torch.equal(torch_before, torch.get_rng_state()))
                            parity.append({'model': model, 'seed': seed, 'aid': aid, 'passed': passed,
                                           'steps': len(tape), 'action_tape_sha256': hashlib.sha256(pickle.dumps(tape)).hexdigest()})
                            dump(output / 'parity.json', parity)
                            if not passed:
                                raise AssertionError('Trace changed policy actions, outcomes or RNG')
                        game_id = f'{model}-{seed}-A{aid}'
                        for row in trace:
                            row.update(model=model, seed=seed, aid=aid, game_id=game_id)
                            if row['active_plan_id']:
                                row['active_plan_id'] = model + '-' + row['active_plan_id']
                        for p in starts:
                            p.update(model=model, seed=seed, aid=aid, game_id=game_id)
                            p['plan_id'] = model + '-' + p['plan_id']
                        for c in branches:
                            c.update(model=model, game_id=game_id)
                        games.append(result)
                        diagnostics.extend(diagnostic_row(r) for r in trace)
                        trackers.extend(starts)
                        checks.extend(branches)
                        categories = replay_categories(result, trace, starts, branches)
                        case = {'game_id': game_id, 'categories': categories, 'result': result, 'timeline': trace}
                        if seed == args.seed:
                            pool[('coverage_example', model, aid)] = case
                        for category in categories:
                            pool.setdefault((category, model, aid), case)
                        if len({c['game_id'] for c in pool.values()}) < 10:
                            pool[('fallback', game_id, aid)] = case
                        writing = perf_counter()
                        for row in trace:
                            trace_file.write(json.dumps(row, ensure_ascii=False) + '\n')
                        for row in branches:
                            check_file.write(json.dumps(row, ensure_ascii=False) + '\n')
                        for row in starts:
                            plan_file.write(json.dumps(row, ensure_ascii=False) + '\n')
                        game_file.write(json.dumps({'model': model, 'game_id': game_id, **result}) + '\n')
                        for stream in (trace_file, check_file, plan_file, game_file):
                            stream.flush()
                        write_seconds += perf_counter() - writing
                        completed_games += 1
                        dump(output / 'status.json', {'status': 'running', 'completed_games': completed_games,
                                                     'expected_games': 4 * args.boards, 'elapsed_seconds': perf_counter() - started})
                        print(f'[{completed_games}/{4 * args.boards}] {game_id} steps={result["steps"]} '
                              f'labels={sum(bool(r["first_actions"]) for r in trace)} '
                              f'clone_states={len(branches)//6} elapsed={(perf_counter()-started)/60:.1f} min', flush=True)
                all_games[model] = games
                models[model] = {'overall': summarize(games, diagnostics, trackers, checks),
                                 'by_corner': {str(a): summarize([g for g in games if g['aid'] == a],
                                     [d for d in diagnostics if d['aid'] == a],
                                     [p for p in trackers if p['aid'] == a], [c for c in checks if c['aid'] == a]) for a in (0, 1)}}
        cases = select_replays(pool, 10)
        write_replays(output, cases)
        after = {name: digest(path) for name, path in paths.items()}
        if after != hashes:
            raise RuntimeError('Sources/checkpoints changed during audit; results are invalid')
        from pilot_two_bomb import paired_active_bootstrap
        difference = paired_active_bootstrap({'games': all_games['two_bomb_final']}, {'games': all_games['original']})
        summary = {'status': 'complete', 'decision': 'review_pending_keep_original', 'models': models,
                   'best_alias': 'original', 'games_completed': completed_games, 'replay_cases': len(cases),
                   'parity_passed': all(p['passed'] for p in parity), 'hashes_unchanged': True,
                   'full_protocol': config['full_protocol'], 'final_minus_original_activewr': difference,
                   'elapsed_seconds': perf_counter() - started, 'trace_write_seconds': write_seconds,
                   'limitations': [
                       'Fixed-policy audit: không tái tạo các state/policy thay đổi trong train 200 episode.',
                       'Chỉ clone các state sampled theo config; đối thủ clone là fixed policy phản ứng, không fixed position/tape (trừ static).',
                       'Masked check chỉ kiểm tra prefix tới bom cuối. Suffix là timed BFS vật lý dựa trên bom đã biết, không CNN hay safety mask.',
                       'Suffix chưa dự báo các bom đối thủ đặt thêm; full-horizon alive chỉ là kết quả của branch đã chạy.',
                       'Post-terminal physics tách riêng, không dùng làm ActiveWR/owned kill thật của trận CNN.',
                       'Churn event association là proxy; unknown chưa đủ chứng minh đổi kế hoạch không do state mới.',
                       'Tied first actions đều được giữ; completion theo một path tương ứng, strict path khác có thể vẫn hợp lý.',
                       'Timing teacher đã gồm observer; không so trực tiếp với timing train T1. Kill trong plan window chưa có causal attribution.',
                       'Best two_bomb bằng original ở từng tensor nên alias, không chạy lại trận giống hệt.',
                       'Audit set là exploratory; không dùng để chọn/promote checkpoint hoặc tune rồi báo như test độc lập.']}
        dump(output / 'summary.json', summary)
        write_report(output, summary)
        dump(output / 'status.json', {'status': 'complete', 'completed_games': completed_games,
                                     'expected_games': 4 * args.boards, 'full_protocol': config['full_protocol'],
                                     'elapsed_seconds': perf_counter() - started})
        archive = shutil.make_archive(str(output) + '-send', 'zip', root_dir=output)
        print(f'COMPLETE. Report: {output / "report.md"}\nSend: {archive}', flush=True)
        return output
    except BaseException as exc:
        dump(output / 'status.json', {'status': 'incomplete', 'completed_games': completed_games,
                                     'expected_games': 4 * args.boards, 'error': f'{type(exc).__name__}: {exc}',
                                     'elapsed_seconds': perf_counter() - started})
        print(f'INCOMPLETE. Keep folder for diagnosis: {output}', flush=True)
        raise


if __name__ == '__main__':
    main()
