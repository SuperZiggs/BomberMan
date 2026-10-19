"""SD7.9c: G+search2 (two-move kill-search) vs G+search, test seed 47000, once. No training.

Opened because SD7.9b validation passed its pre-set rule (+17.5pp >= +2pp). Gate (schedule E), as SD7.8:
G+search2 - G+search ActiveWR >= +5pp, CI95 lower > 0, suicide <= +1pp. Schedule off is report-only.
"""
import argparse
import json
from pathlib import Path
import shutil

from progress import parallel_map
import run_sd76_pilot as sd76
import run_sd5_pilot as sd5
from run_sd78_search_test import gate, summarize
from run_sd79_probe import play

ARMS, RULES, SEED, GAMES, MAX_STEPS, SHARDS = ('G_search', 'G_search2'), ('E', 'off'), 47000, 400, 500, 4


def main(output, workers=5, smoke=False):
    from pilot_two_bomb import paired_active_bootstrap
    if sd5._hash(sd76.G) != sd76.G_SHA256:
        raise ValueError('G checkpoint changed')
    output = Path(output)
    output.mkdir(parents=True)  # FileExistsError: never overwrite
    seed, games, steps, shards = (21000, 4, 40, 2) if smoke else (SEED, GAMES, MAX_STEPS, SHARDS)
    with open(output / 'test_opened.json', 'x', encoding='utf-8') as handle:  # the test seed is used once
        json.dump({'seed': seed, 'games': games, 'G_sha256': sd76.G_SHA256}, handle, indent=2)
    jobs = [(arm, seed + i * games // 2 // shards, games // shards, steps, rule)
            for arm in ARMS for rule in RULES for i in range(shards)]
    parts = parallel_map(play, jobs, total=len(ARMS) * len(RULES) * games, desc=f'SD7.9 test seed {seed}', workers=workers)
    results = {f'{a}_{r}': sorted([g for pa, pr, gs in parts if (pa, pr) == (a, r) for g in gs],
                                  key=lambda g: (g['seed'], g['aid'])) for a in ARMS for r in RULES}
    for key, rows in results.items():
        (output / f'evaluation_{key}_{seed}.json').write_text(json.dumps({'games': rows}), encoding='utf-8')
    paired = {rule: {m: paired_active_bootstrap(*({'games': [{**g, 'active_win': g[m]} for g in results[f'{a}_{rule}']]}
                                                  for a in ('G_search2', 'G_search')))
                     for m in ('active_win', 'win', 'draw', 'passive_win', 'suicide')} for rule in RULES}
    summary = {'seed': seed, 'games': games, 'results': {k: summarize(v) for k, v in results.items()},
               'search2_vs_search': paired, 'gate': gate(paired['E'])}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not smoke:
        shutil.make_archive(str(output) + '-send', 'zip', output, '.')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    main(a.output, a.workers, a.smoke)
