"""Opponent-aware escape (SD6): safe moves that leave an opponent no way out of a blast.

F almost never plays these (probability ~0.001 at the moves kill-search found, RESULTS SD6.0):
the escape teacher always labels the nearest refuge and ignores the opponent.

shortcut: the learner is assumed to hold the new cell until the blast and the opponent to
escape by walking only (no counter-bomb); kill-search (`kill_search.py`) is the reference.
"""
from bomberman_env import ACTION_STOP, _MOVE_DELTA
from rule_based_bot import _survival_map


def _opponent_survives(env, opponent):
    safe, _, _ = _survival_map(env, opponent.row, opponent.col, avoid_contests=False)
    return bool(safe[0, opponent.row, opponent.col])


def trap_actions(env, aid, mask):
    """Mask-allowed STOP/moves to a cell that never burns and from which the learner, holding it,
    leaves some opponent no escape, while another allowed STOP/move would let that opponent out.
    Usually STOP in the opponent's last doorway (RESULTS SD6.1: F runs away and frees it)."""
    if not env.bombs:
        return []
    me = env.agents[aid]
    origin = (me.row, me.col)
    opponents = [a for a in env.agents if a.alive and a.agent_id != aid]
    _, _, burns = _survival_map(env, *origin, avoid_contests=False)
    doomed, holdable = {}, set()  # allowed STOP/move -> opponents left with no escape
    try:  # the learner is moved in place and restored: deepcopy per candidate is too slow
        for action, (dr, dc) in {ACTION_STOP: (0, 0), **_MOVE_DELTA}.items():
            if not mask[action]:
                continue
            me.row, me.col = origin[0] + dr, origin[1] + dc
            doomed[action] = {a.agent_id for a in opponents if not _opponent_survives(env, a)}
            if not burns[:, me.row, me.col].any():
                holdable.add(action)
    finally:
        me.row, me.col = origin
    if not doomed:
        return []
    always = set.intersection(*doomed.values())  # doomed whatever the learner does
    return [action for action in doomed if action in holdable and doomed[action] - always]
