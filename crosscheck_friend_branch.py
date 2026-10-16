"""Cross-check of codex/phasec-sd-best-fallback (a6650a0): is its 67.5% WR / 51% ActiveWR real?

Runs the branch's OWN evaluate.py (its env, 13-channel obs, SD-aware RuleBot) from a separate
git worktree, for its checkpoint and for B (same warm-start file 93bdf278...), then pairs the
games with our bootstrap. B's 12 channels are zero-extended by the branch loader (B unchanged).

  E1  friend, schedule E (SD 475/25/40), seed 31000, 200 games: must reproduce 67.5 / 51.0
  E2  B, same as E1: control on the friend's own test boards
  E3  friend vs B, schedule E, fresh cross-check seed 60000, 400 games each
  E4  friend vs B, sudden death OFF, seed 60000, 400 games each (plain game, no ring)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from pilot_two_bomb import paired_active_bootstrap

ROOT = Path(__file__).resolve().parent
PYTHON = r'D:\anaconda\envs\torch\python.exe'
B = ROOT / 'runs/pressure-C-next/final_phaseC.pt'
FRIEND = 'runs/phaseC-sd-best-fallback/best_phaseC.pt'
SCHEDULE_E = ['--sudden_death', '--sudden_death_start', '475', '--sudden_death_interval', '25',
              '--sudden_death_warning_steps', '40']
JOBS = {  # name: (model, seed, games, schedule E?)
    'E1_friend_E_31000': ('friend', 31000, 200, True), 'E2_B_E_31000': ('B', 31000, 200, True),
    'E3_friend_E_60000': ('friend', 60000, 400, True), 'E3_B_E_60000': ('B', 60000, 400, True),
    'E4_friend_off_60000': ('friend', 60000, 400, False), 'E4_B_off_60000': ('B', 60000, 400, False)}
PAIRS = {'E2_friend_vs_B_E_31000': ('E1_friend_E_31000', 'E2_B_E_31000'),
         'E3_friend_vs_B_E_60000': ('E3_friend_E_60000', 'E3_B_E_60000'),
         'E4_friend_vs_B_off_60000': ('E4_friend_off_60000', 'E4_B_off_60000')}
METRICS = ('win', 'active_win', 'draw', 'suicide')
KEYS = ('win_rate', 'active_win_rate', 'passive_win_rate', 'sudden_death_win_rate', 'draw_rate',
        'suicide_rate', 'sudden_death_death_rate', 'avg_kills', 'avg_steps')


def run(name, worktree, output):
    model, seed, games, sd = JOBS[name]
    path = output / f'evaluation_{name}.json'
    if not path.exists():
        command = [PYTHON, 'evaluate.py', '--mode', 'seeded', '--board_size', '9', '--num_agents', '2',
                   '--num_episodes', str(games), '--max_steps', '500', '--seed', str(seed), '--device', 'cpu',
                   '--ac_model', str(B) if model == 'B' else FRIEND, '--output', str(path),
                   *(SCHEDULE_E if sd else [])]
        subprocess.run(command, cwd=worktree, check=True, stdout=subprocess.DEVNULL)
    result = json.loads(path.read_text(encoding='utf-8'))
    print(name, {k: result.get(k) for k in KEYS}, flush=True)
    return name, result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worktree', required=True, type=Path, help='checkout of a6650a0 with LFS files')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'runs/crosscheck-friend-a6650a0')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)  # finished evaluations are reused on rerun
    with ThreadPoolExecutor(a.workers) as pool:
        results = dict(pool.map(lambda n: run(n, a.worktree, a.output_dir), JOBS))
    summary = {'branch_commit': 'a6650a0', 'results': {n: {k: r.get(k) for k in KEYS} for n, r in results.items()},
               'paired': {}}
    for label, (left, right) in PAIRS.items():
        summary['paired'][label] = {}
        for metric in METRICS:
            wrap = lambda r: {**r, 'games': [{**g, 'active_win': g[metric]} for g in r['games']]}
            summary['paired'][label][metric] = paired_active_bootstrap(wrap(results[left]), wrap(results[right]))
    (a.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for label, metrics in summary['paired'].items():
        print(label, {m: (round(v['difference_pp'], 2), v['ci95_pp']) for m, v in metrics.items()}, flush=True)
