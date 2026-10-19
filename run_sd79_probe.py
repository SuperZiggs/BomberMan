"""SD7.9: is SD5.3's CNN (S) a better base than G for play-time kill-search? Validation only.

Stage b (SD7.9b, rule fixed after stage a, before running b): G+search2 (two-move search,
`kill_search.search_action2`) vs stage a's G+search games on the same boards; if ActiveWR
>= +2pp, G+search2 is the candidate for the one-shot test on seed 47000.

Arms on validation seed 21000, 400 games each, schedule E: G+search (current best agent), S, S+search.
Rule fixed before running: if S+search - G+search ActiveWR >= +2pp, S+search becomes the candidate
for a one-shot test on seed 47000 (gate +5pp, CI lower > 0, suicide <= +1pp); otherwise no test seed
is spent. S alone vs G+search is report-only (a search-free agent at no play-time cost).
"""
import argparse
import hashlib
import json
from pathlib import Path

from progress import parallel_map
import run_sd5_pilot as sd5
import run_sd76_pilot as sd76
from run_sd78_search_test import summarize

S = Path(__file__).resolve().parent / 'runs/sd53-20261018/train/epoch_8.pt'
S_SHA256 = '8f7116d290593fa10de1753c4f0f96366e220a193e9860123ed72e0db46bfd4a'
ARMS = {'G_search': (sd76.G, 1), 'S': (S, 0), 'S_search': (S, 1), 'G_search2': (sd76.G, 2)}
STAGES = {'a': ('G_search', 'S', 'S_search'), 'b': ('G_search2',)}
SEED, GAMES, MAX_STEPS, SHARDS = 21000, 400, 500, 4


def play(arm, seed, games, max_steps, rule, ticks):
    import torch
    torch.set_num_threads(1)
    import evaluate
    import train_actor_critic as trainer
    from kill_search import search_action, search_action2
    checkpoint, mode = ARMS[arm]
    search = {1: search_action, 2: search_action2}.get(mode)
    policy = evaluate.make_ac_fn(str(checkpoint), 9)
    agent = policy if not search else (
        lambda env, obs, aid: search(env, aid, trainer.get_action_mask(env, aid), policy(env, obs, aid))[0])
    done = [0]

    def advance(value, total):
        ticks.put(value - done[0]); done[0] = value
    result = evaluate.evaluate_seeded(agent, 9, 2, games, seed, max_steps, on_progress=advance,
                                      sudden_death=rule == 'E', **sd5.SCHEDULE_E)
    return arm, rule, result['games']


def decide(paired, arm='S_search'):
    return {f'{arm}_minus_G_search': paired[arm]['active_win']['difference_pp'],
            'open_test_47000': paired[arm]['active_win']['difference_pp'] >= 2.0}


def main(output, workers=5, smoke=False, stage='a', base=None):
    from pilot_two_bomb import paired_active_bootstrap
    for path, sha in ((sd76.G, sd76.G_SHA256), (S, S_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f'checkpoint changed: {path}')
    output = Path(output)
    output.mkdir(parents=True)  # FileExistsError: never overwrite
    games, steps = (4, 40) if smoke else (GAMES, MAX_STEPS)
    shards = 2 if smoke else SHARDS
    jobs = [(arm, SEED + i * games // 2 // n, games // n, steps, 'E')
            for arm in STAGES[stage] for n in [shards if ARMS[arm][1] else 1] for i in range(n)]
    parts = parallel_map(play, jobs, total=len(STAGES[stage]) * games, desc=f'SD7.9 validation {SEED}', workers=workers)
    if stage == 'b':  # reuse stage a's G+search games: same machine, same boards
        parts = parts + [('G_search', 'E', json.loads((Path(base) / f'evaluation_G_search_{SEED}.json').read_text(encoding='utf-8'))['games'])]
    results = {arm: sorted([g for a, _, gs in parts if a == arm for g in gs], key=lambda g: (g['seed'], g['aid']))
               for arm in {*STAGES[stage], 'G_search'}}
    for arm, rows in results.items():
        (output / f'evaluation_{arm}_{SEED}.json').write_text(json.dumps({'games': rows}), encoding='utf-8')
    paired = {arm: {m: paired_active_bootstrap(*({'games': [{**g, 'active_win': g[m]} for g in results[a]]}
                                                 for a in (arm, 'G_search')))
                    for m in ('active_win', 'win', 'draw', 'passive_win', 'suicide')} for arm in STAGES[stage] if arm != 'G_search'}
    summary = {'seed': SEED, 'games': games, 'results': {k: summarize(v) for k, v in results.items()},
               'vs_g_search': paired, 'decision': decide(paired, 'S_search' if stage == 'a' else 'G_search2')}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stage', choices=STAGES, default='a')
    p.add_argument('--base', type=Path, help='stage b: stage-a output holding the G+search games')
    a = p.parse_args()
    main(a.output, a.workers, a.smoke, a.stage, a.base)
