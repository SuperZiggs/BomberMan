"""Copy engine data into JSON values. No renderer or simulation mutation."""
from dataclasses import asdict

from bomberman_env import ACTION_STOP
from rule_based_bot import _build_danger_map


def capture_state(env, labels, actions=None, rewards=None):
    alive = [a.agent_id for a in env.agents if a.alive]
    result = None
    if env._episode_done:
        winner = alive[0] if len(alive) == 1 else None
        result = {"winner": winner, "outcome": "win" if winner is not None else "draw",
                  "title": f"A{winner} · {labels[winner]} thắng" if winner is not None else "Ván đấu hòa"}
    closure = env._next_sudden_death_closure() if env.sudden_death else None
    remaining = closure[0] if closure else None
    warning = closure is not None and 0 < remaining <= env.sudden_death_warning_steps
    return {
        "grid": env.grid.tolist(), "agents": [asdict(a) for a in env.agents],
        "bombs": [asdict(b) for b in env.bombs],
        "flames": [list(pos) for pos in env.flames],
        "powerups": [[list(pos), kind] for pos, kind in env.powerups.items()],
        "danger": _build_danger_map(env).tolist(),
        "ring": {"enabled": bool(env.sudden_death), "remaining": remaining if warning else None,
                 "cells": [list(p) for p in sorted(closure[1])] if warning else []},
        "step": env.step_count, "tick": env.step_count, "max_steps": env.max_steps,
        "done": bool(env._episode_done), "result": result, "labels": list(labels),
        "actions": list(actions) if actions is not None else [ACTION_STOP] * env.num_agents,
        "rewards": [float(r) for r in rewards] if rewards is not None else [0.] * env.num_agents,
    }
