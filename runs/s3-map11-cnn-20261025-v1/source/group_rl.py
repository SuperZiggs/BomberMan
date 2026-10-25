"""SD7: group-relative policy gradient from decisive states.

A decisive state is one where `kill_search` finds a learner-credited kill within 12 ticks
(SD7.0 spike: F converts 60% of them, 86% are mixed over 8 replays). From each such state the
current policy replays K times for H ticks against fresh RuleBots; advantage = outcome - group
mean, so only the learner's choices separate the replays and no value function is needed.

shortcut: opponents are always RuleBot and the outcome window is H ticks; for self-play
opponents or longer plans, replay against the episode's own opponents instead.
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from bomberman_env import ACTION_BOMB, ACTION_STOP
from rule_based_bot import RuleBasedAgent, _safe_action_mask

K, HORIZON, MAX_STATES, GAP, REACH = 8, 20, 3, 8, 4


def decisive(env, aid, mask):
    """Cheap filter (live bomb, opponent in danger and near), then kill-search."""
    me = env.agents[aid]
    if not (me.alive and env.bombs):
        return False
    danger = env._danger_map()
    if not any(a.alive and a.agent_id != aid and danger[a.row, a.col] > 0
               and abs(a.row - me.row) + abs(a.col - me.col) <= REACH for a in env.agents):
        return False
    from kill_search import search_action
    _, info = search_action(env, aid, mask, ACTION_STOP)
    return not info['skipped'] and max(info['scores'].values(), default=0.0) == 1.0


def kill_actions(env, aid, mask):
    """Actions kill-search scores as a learner-credited kill; [] unless the bomb is legal and an
    opponent is within REACH (the search costs ~19 ms, so it is skipped elsewhere)."""
    me = env.agents[aid]
    if not (me.alive and mask[ACTION_BOMB]) or not any(
            a.alive and a.agent_id != aid and abs(a.row - me.row) + abs(a.col - me.col) <= REACH for a in env.agents):
        return []
    import kill_search
    _, info = kill_search.search_action(env, aid, mask, ACTION_STOP)
    return [] if info['skipped'] else [a for a, s in info['scores'].items() if s == 1.0]


def narrow(env, aid, mask, max_kill=2):
    """SD7.6: a "now or never" moment with only 1..max_kill killing actions
    (SD7.6a: 31/37 failed kill windows last one step)."""
    return 1 <= len(kill_actions(env, aid, mask)) <= max_kill


def rollout(env0, aid, select, seed, horizon=HORIZON):
    """Replay a copy for up to `horizon` ticks; select(obs, mask) -> learner action.
    Returns (+1 learner's bomb kills while it lives / -1 learner dies / 0, [(obs, mask, action)])."""
    env = deepcopy(env0)
    bots = {a.agent_id: RuleBasedAgent(a.agent_id) for a in env.agents if a.agent_id != aid}
    state = np.random.get_state()  # RuleBot's random fallback must not shift the real game
    np.random.seed(seed)
    steps = []
    try:
        obs = env._get_obs(aid)
        for _ in range(horizon):
            actions = [ACTION_STOP] * env.num_agents
            for other, bot in bots.items():
                if env.agents[other].alive:
                    actions[other] = bot.select_action(env)
            mask = _safe_action_mask(env, aid)
            actions[aid] = select(obs, mask)
            steps.append((obs, mask, actions[aid]))
            obs_list, _, _, info = env.step(actions)
            obs = obs_list[aid]
            if any(k == aid and v != aid for k, v in info['kills']) and env.agents[aid].alive:
                return 1, steps
            if not env.agents[aid].alive:
                return -1, steps
            if info['episode_done']:
                return 0, steps
        return 0, steps
    finally:
        np.random.set_state(state)


def sampler(net, device, seed):
    """Masked sampling from the current policy with its own generator (the trainer's RNG is untouched)."""
    generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def select(obs, mask):
        logits, _ = net(torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device))
        logits = logits.masked_fill(~torch.tensor(mask, dtype=torch.bool, device=device), -1e9)
        return int(torch.multinomial(torch.softmax(logits, -1)[0].cpu(), 1, generator=generator))
    return select


def group_advantages(outcomes):
    mean = sum(outcomes) / len(outcomes)
    return [float(o - mean) for o in outcomes]


def group_loss(net, groups, device, first_only=False):
    """groups: [[(outcome, steps), ...], ...]; mean of -A * log pi over every replayed learner step
    (first_only: only the action taken at the stored state, SD7.6)."""
    obs, masks, actions, weights = [], [], [], []
    stats = dict(groups=len(groups), contrast_groups=0, rollouts=0, wins=0, deaths=0, steps=0)
    for group in groups:
        outcomes = [o for o, _ in group]
        stats['rollouts'] += len(group)
        stats['wins'] += sum(o == 1 for o in outcomes)
        stats['deaths'] += sum(o == -1 for o in outcomes)
        if len(set(outcomes)) < 2:
            continue  # no contrast, no signal
        stats['contrast_groups'] += 1
        for advantage, (_, steps) in zip(group_advantages(outcomes), group):
            for o, m, a in (steps[:1] if first_only else steps):
                obs.append(o), masks.append(m), actions.append(a), weights.append(advantage)
    stats['steps'] = len(actions)
    if not actions:
        return torch.zeros((), device=device), stats
    logits, _ = net(torch.from_numpy(np.stack(obs).astype(np.float32)).to(device))
    logits = logits.masked_fill(~torch.tensor(masks, dtype=torch.bool, device=device), -1e9)
    log_probs = F.log_softmax(logits, -1).gather(1, torch.tensor(actions, device=device)[:, None])[:, 0]
    return -(torch.tensor(weights, device=device) * log_probs).mean(), stats
