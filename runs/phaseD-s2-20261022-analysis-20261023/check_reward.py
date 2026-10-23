"""Pure event arithmetic from frozen engine rules; no environment rollout."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'phaseD-s2-20261022/sources'))
from train_actor_critic import _shaped_reward, _reward_components

base = dict(kills=[], self_blast_deaths=[], alive=[True, False], episode_done=True,
            sudden_death_deaths=[], walls_destroyed={}, items_collected={0: [], 1: []})
cases = [
    ('ordinary_passive_win', 9.999, base, 0, -0.001),
    ('sudden_death_passive_win', -0.001, {**base, 'sudden_death_deaths': [1]}, 0, -0.001),
    ('owned_kill_active_win', 10.999, {**base, 'kills': [(0, 1)]}, 1, 13.999),
]
results = []
for name, raw, info, kills, expected in cases:
    shaped = _shaped_reward(raw, 0, info, kills)
    assert abs(shaped - expected) < 1e-10
    try:
        _reward_components(raw, 0, info, kills, True)
        diagnostic = 'consistent'
    except ValueError as error:
        diagnostic = str(error)
    results.append(dict(event=name, raw_env_reward=raw, shaped_reward=shaped,
                        active_win=int(kills > 0), reward_diagnostics=diagnostic))
payload = dict(
    hypothesis='Sole survival through sudden death earns the +10 win bonus',
    verdict='REJECTED: frozen engine excludes +10 on a step with sudden-death deaths',
    scope='Synthetic event arithmetic consistent with frozen bomberman_env.py:316, not new games',
    cases=results,
    incidental_finding='Optional reward-component diagnostics omit the engine sudden-death exception; flag was false in this run.',
)
(HERE / 'reward_probe.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
print(json.dumps(payload, indent=2))
