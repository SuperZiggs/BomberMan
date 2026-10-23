"""SDM probe: does G + three-move kill-search draw less against CNN opponents if the search models
the opponent with the opponent's own CNN (argmax of its masked policy) instead of RuleBot?

Follows PDE (seed 50000): there G + search3 vs frozen CNNs drew 50-70% of games, mostly at 500 steps.
Validation seed 21000 by default (not a test seed). 1v1 9x9, schedule E, max_steps 500, the real
opponent samples its policy; the search model is its argmax (known mismatch). Paired by board.
Arms: for each opponent, model 'rulebot' (old search), 'oracle' (opponent's own CNN) or 'self' (always G's CNN).
SDM2 (spec 2026-10-22): --models rulebot self --opponents rulebot S F --reuse runs/sdm-21000.
"""
import argparse
import json
import shutil
from pathlib import Path
import time

from progress import parallel_map
import run_pde_check as pde
import run_sd5_pilot as sd5
from run_sd78_search_test import summarize

OPPONENTS = ('S', 'G')
MODELS = ('rulebot', 'oracle')


def model_ckpt(model, opponent):
    """Checkpoint the search uses to predict the opponent; None = RuleBot."""
    return {'rulebot': None, 'oracle': pde.CKPT.get(opponent), 'self': pde.CKPT['G']}[model]
METRICS = pde.METRICS


def cnn_model(path, board):
    """Opponent model for kill_search: argmax of the checkpoint's masked policy (deterministic, no RNG)."""
    import torch
    from bomberman_env import OBS_CHANNELS
    from train_actor_critic import ActorCriticNet, get_action_mask, obs_to_tensor, smart_load_weights
    weights = torch.load(str(path), map_location='cpu', weights_only=True)
    channels = max(weights['conv.0.weight'].shape[1], OBS_CHANNELS)
    net = ActorCriticNet(in_channels=channels, board_size=board)
    smart_load_weights(net, str(path), 'cpu')
    net.eval()

    @torch.no_grad()
    def fn(env, aid):
        logits, _ = net(obs_to_tensor(env._get_obs(aid)[:channels], 'cpu'))
        mask = torch.tensor(get_action_mask(env, aid), dtype=torch.bool)
        return int(logits.reshape(-1).masked_fill(~mask, -1e9).argmax())
    return fn


def play(opponent, model, seed, boards, max_steps, shard_path, ticks):
    import torch
    torch.set_num_threads(1)
    import evaluate
    import kill_search
    ckpt = model_ckpt(model, opponent)
    kill_search.OPPONENT_MODEL = None if ckpt is None else cnn_model(ckpt, 9)
    done = [0]

    def advance(value, total):
        ticks.put(value - done[0]); done[0] = value
    start = time.time()
    result = evaluate.evaluate_seeded(pde._seat('G_search3', 9)(0), 9, 2, boards * 2, seed, max_steps,
                                      on_progress=advance, sudden_death=True, opponent_fn=pde._seat(opponent, 9),
                                      **sd5.SCHEDULE_E)
    part = (f'{model}_vs_{opponent}', time.time() - start, result['games'])
    Path(shard_path).write_text(json.dumps(part), encoding='utf-8')
    return part


def run(output, seed, boards, workers, max_steps=500, shard=5, resume=False, opponents=OPPONENTS, models=MODELS,
        reuse=None):
    if sd5._hash(pde.CKPT['G']) != pde.sd76.G_SHA256:
        raise ValueError('G checkpoint changed')
    if models[0] != 'rulebot' or any(m == 'oracle' and o == 'rulebot' for m in models for o in opponents):
        raise ValueError("models must start with 'rulebot' (the reference); oracle needs a CNN opponent")
    output = Path(output)
    opened = {'seed': seed, 'boards': boards, 'max_steps': max_steps, 'opponents': opponents, 'models': models,
              'learner': 'G + search_action3', 'G_sha256': pde.sd76.G_SHA256}
    if (opponents, models) != (OPPONENTS, MODELS):  # SDM's opened.json had no hash; keep it resumable
        opened['kill_search_sha256'] = sd5._hash(Path(__file__).with_name('kill_search.py'))
        opened['reuse'] = str(reuse) if reuse else None
    if resume:
        if json.loads((output / 'opened.json').read_text(encoding='utf-8')) != json.loads(json.dumps(opened)):
            raise ValueError('registered design differs from this run; refusing to resume')
    else:
        output.mkdir(parents=True)  # FileExistsError: never overwrite
        (output / 'opened.json').write_text(json.dumps(opened, indent=2), encoding='utf-8')
    (output / 'shards').mkdir(exist_ok=True)
    jobs = [(o, m, seed + first, min(shard, boards - first), max_steps,
             str(output / 'shards' / f'{m}_vs_{o}_{seed + first}.json'))
            for o in opponents for m in models for first in range(0, boards, shard)]
    for j in jobs:  # same seed/boards/code: finished shards from an earlier probe are reused, not rerun
        old = Path(reuse or '') / 'shards' / Path(j[-1]).name
        if reuse and old.exists() and not Path(j[-1]).exists():
            shutil.copy(old, j[-1])
    done = [j for j in jobs if Path(j[-1]).exists()]
    todo = sorted((j for j in jobs if j not in done), key=lambda j: j[1] == 'rulebot')  # slower CNN models first
    parts = [tuple(json.loads(Path(j[-1]).read_text(encoding='utf-8'))) for j in done]
    parts += parallel_map(play, todo, total=sum(j[3] * 2 for j in todo), desc=f'SDM seed {seed}', workers=workers)
    results, seconds = {}, {}
    for name, sec, games in parts:
        results.setdefault(name, []).extend(games); seconds[name] = seconds.get(name, 0) + sec
    for rows in results.values():
        rows.sort(key=lambda g: (g['seed'], g['aid']))
    summary = {**opened, 'schedule': sd5.SCHEDULE_E,
               'seconds_per_game': {k: v / len(results[k]) for k, v in seconds.items()},
               'results': {k: {**summarize(v), 'loss_rate': 100 * sum(not g['win'] and not g['draw'] for g in v) / len(v),
                               'draw_at_max_steps': 100 * sum(g['draw'] and g['steps'] >= max_steps for g in v) / len(v)}
                           for k, v in results.items()},
               'comparisons': {f'{c} - rulebot vs {o}': {
                   m: pde.paired_bootstrap([{**g, 'value': g[m]} for g in results[f'{c}_vs_{o}']],
                                           [{**g, 'value': g[m]} for g in results[f'rulebot_vs_{o}']])
                   for m in METRICS} for o in opponents for c in models[1:]}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for k, v in sorted(summary['results'].items()):
        print(f"{k:16s} ActiveWR {v['active_win_rate']:6.2f} WR {v['win_rate']:6.2f} draw {v['draw_rate']:6.2f} "
              f"loss {v['loss_rate']:6.2f} s/game {summary['seconds_per_game'][k]:6.1f}", flush=True)
    for k, c in summary['comparisons'].items():
        a = c['active_win']
        print(f"{k}: ActiveWR {a['difference_pp']:+.2f} [{a['ci95_pp'][0]:+.2f}, {a['ci95_pp'][1]:+.2f}], "
              f"draw {c['draw']['difference_pp']:+.2f}", flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=21000)
    p.add_argument('--boards', type=int, default=50)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--max_steps', type=int, default=500)
    p.add_argument('--resume', action='store_true', help='keep finished shards in OUTPUT/shards, run the rest')
    p.add_argument('--opponents', nargs='+', default=list(OPPONENTS), help="'rulebot' or a run_pde_check.CKPT key")
    p.add_argument('--models', nargs='+', default=list(MODELS), choices=('rulebot', 'oracle', 'self'))
    p.add_argument('--reuse', type=Path, help='earlier probe output whose matching shards are copied, not rerun')
    a = p.parse_args()
    run(a.output, a.seed, a.boards, a.workers, a.max_steps, resume=a.resume,
        opponents=tuple(a.opponents), models=tuple(a.models), reuse=a.reuse)
