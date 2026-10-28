"""SD9-SEEDS: two more independent replicates of the SD9 distillation round (S2 + search3 -> S3), to measure
seed variance that the paper's single-seed result cannot show. Descriptive; nothing is selected after the test.

Each replicate repeats SD9 exactly (same S2 start, teacher, 3000 games, schedule E p 0.75, anchors 5%, lambda 1,
Adam 1e-4, batch 256, 16 epochs, candidates 4/8/12/16, validation seed 21000, same selection rule), with
fresh collection boards and its own train seed (batch order and held-out split):
  r1: boards 230000-231499, train seed 531
  r2: boards 232000-233499, train seed 532
Fixed before running: one common fresh test block, seed 61000 (200 boards x 2 corners, schedule E, used once),
plays the original S3 (SD9, seed 530), S3-r1, S3-r2 and S2. Reported: each S3 - S2 paired ActiveWR with CI,
whether each passes the original SD9 gate (>= +5pp, CI lower > 0, suicide <= +1pp), and mean / min / max of
S3 - S2 over the three seeds.

  collect  --rep r1|r2 [--workers 3] [--resume]
  train    --rep r1|r2
  validate --rep r1|r2
  test     (after both replicates are validated; once)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from progress import bar, parallel_map
import run_sd53_pilot as sd53
import run_sd9_distill as sd9

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'runs/sd9-seeds-20261026'
REPS = {'r1': (230000, 231499, 531), 'r2': (232000, 233499, 532)}
TEST_SEED = 61000
S3_ORIGINAL = ROOT / 'runs/sd9-20261023/train/epoch_12.pt'
S3_SHA256 = '4185375f8b5f5cf070d3eef6ce98bec27e23fe06c64d97b500c939e36e669ed7'


def config_for(rep, smoke=False):
    first, last, seed = REPS[rep]
    boards = list(range(first, last + 1))[:4 if smoke else None]
    out = OUTPUT.parent / (OUTPUT.name + '-smoke') if smoke else OUTPUT
    return {**sd9.config_for(out / rep, smoke), 'protocol': 'SD9-SEEDS replicate of SD9', 'replicate': rep,
            'collect_board_seeds': boards, 'collect_games': len(boards) * 2, 'train_seed': seed,
            'test_seed': TEST_SEED}


def _sources():
    return {**sd9._sources(), 'run_sd9_seeds.py': sd53.normalized_sha256(Path(__file__))}


def _registration(output, config, create=False):
    marker = Path(output) / 'registration.json'
    if create:
        Path(output).mkdir(parents=True)  # FileExistsError: never overwrite a run
        sd53._atomic_json(marker, {'config': config, 'sources_normalized_lf_sha256': _sources(), 'runtime': sd53._runtime()})
        return
    registration = json.loads(marker.read_text(encoding='utf-8'))
    if registration['config'] != config or registration['sources_normalized_lf_sha256'] != _sources():
        raise ValueError('configuration or source differs from registration; refused')


def collect(rep, smoke=False, workers=3, resume=False):
    sd9._require(sd9.S2, sd9.S2_SHA256)
    config = config_for(rep, smoke)
    output = Path(config['output'])
    if output.exists() and not resume:
        raise FileExistsError(f'{output} exists; use --resume only for an interrupted collect')
    _registration(output, config, create=not output.exists())
    (output / 'collection').mkdir(exist_ok=True)
    jobs = []
    for start, end in sd9._shards(config):
        path = output / 'collection' / f'shard-{start}-{end}.npz'
        if path.exists() and not sd9._valid_shard(path, start, end, config):
            raise ValueError(f'invalid shard {path}; delete it explicitly before recollecting')
        if not path.exists():
            jobs.append((start, end, str(output), config))
    if jobs:
        parallel_map(sd9._collect_shard, jobs, total=sum((e - s + 1) * 2 for s, e, *_ in jobs),
                     desc=f'SD9-SEEDS {rep} collect', workers=max(1, min(workers, len(jobs))))
    games, decisions, anchors = [], 0, 0
    for path in sorted((output / 'collection').glob('shard-*.npz')):
        with np.load(path, allow_pickle=False) as data:
            games += json.loads(str(data['games'])); decisions += len(data['decision_seed']); anchors += len(data['anchor_seed'])
    if len(games) != config['collect_games']:
        raise ValueError('collection is incomplete')
    summary = {'games': len(games), 'decisions': decisions, 'anchors': anchors,
               'teacher_rates': {k: 100 * sum(bool(g[k]) for g in games) / len(games) for k in ('win', 'active_win', 'draw')}}
    sd53._atomic_json(output / 'collection_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def train(rep, smoke=False, device='cpu'):
    """sd9.train with the replicate's train seed in place of 530 (held-out split and batch order)."""
    import torch
    import torch.nn.functional as functional
    sd9._require(sd9.S2, sd9.S2_SHA256)
    config = config_for(rep, smoke)
    output, seed = Path(config['output']), config['train_seed']
    _registration(output, config)
    records = sd9._load_records(output, config)
    target = np.stack([sd53.masked_soft_target(s, m) for s, m in zip(records['decision_scores'], records['decision_mask'])])
    held = sd53._heldout_board_ids(np.concatenate((records['decision_seed'], records['anchor_seed'])),
                                   config['heldout_fraction'], seed)
    d_train, d_held = sd53._partition_indices(records['decision_seed'], held)
    a_train, a_held = sd53._partition_indices(records['anchor_seed'], held)
    d_train = np.intersect1d(d_train, np.flatnonzero(target.sum(1) > 0))
    if not (len(d_train) and len(a_train) and len(d_held) and len(a_held)):
        raise ValueError('insufficient records after the board-seed split')
    torch.manual_seed(seed)
    net = sd53._load_net(sd9.S2, device).train()
    optimizer = torch.optim.Adam([p for n, p in net.named_parameters() if not n.startswith('critic_head.')],
                                 lr=config['learning_rate'])
    rng, half = np.random.default_rng(seed), config['batch_size'] // 2
    destination = output / 'train'
    destination.mkdir()  # no silent reuse
    batches = max(-(-len(d_train) // half), -(-len(a_train) // half))
    as_tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)
    metrics = [{'epoch': 0, **sd53._heldout_metrics(net.eval(), records, d_held, a_held, device)}]
    net.train()
    with bar(config['train_epochs'] * batches, f'SD9-SEEDS {rep} train') as shown:
        for epoch in range(1, config['train_epochs'] + 1):
            order_d, order_a, losses = rng.permutation(d_train), rng.permutation(a_train), []
            for batch in range(batches):
                di = order_d[(np.arange(half) + batch * half) % len(order_d)]
                ai = order_a[(np.arange(half) + batch * half) % len(order_a)]
                d_logits, _ = net(as_tensor(records['decision_obs'][di]))
                d_log = functional.log_softmax(sd53._masked_logits(d_logits, as_tensor(records['decision_mask'][di], torch.bool)), 1)
                ce = -(as_tensor(target[di]) * d_log).sum(1).mean()
                a_mask = as_tensor(records['anchor_mask'][ai], torch.bool)
                a_logits, _ = net(as_tensor(records['anchor_obs'][ai]))
                teacher = functional.softmax(sd53._masked_logits(as_tensor(records['anchor_logits'][ai]), a_mask), 1)
                kl = functional.kl_div(functional.log_softmax(sd53._masked_logits(a_logits, a_mask), 1), teacher,
                                       reduction='batchmean')
                loss = ce + config['anchor_lambda'] * kl
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.item())); shown.update(1)
            metrics.append({'epoch': epoch, 'train_loss': float(np.mean(losses)),
                            **sd53._heldout_metrics(net.eval(), records, d_held, a_held, device)})
            net.train()
            if epoch in config['candidate_epochs']:
                torch.save(net.state_dict(), destination / f'epoch_{epoch}.pt')
    sd53._atomic_json(destination / 'metrics.json', {'metrics': metrics})
    print(json.dumps(metrics[-1], indent=2), flush=True)


def validate(rep, smoke=False, workers=5):
    from run_sd78_search_test import summarize
    config = config_for(rep, smoke)
    output = Path(config['output'])
    _registration(output, config)
    if (output / 'selected.json').exists():
        raise FileExistsError('validation already done; selection cannot be overwritten')
    arms = {'S2': (sd9.S2, False, ('E',)), **{f'epoch_{e}': (output / 'train' / f'epoch_{e}.pt', False, ('E',))
                                           for e in config['candidate_epochs']}}
    results = sd9.run_jobs(arms, sd9.VALIDATION_SEED, config['evaluation_games'], sd9.MAX_STEPS, workers,
                           f'SD9-SEEDS {rep} validate')
    summaries = {k.removesuffix('_E'): summarize(v) for k, v in results.items()}
    epoch = sd9.select(summaries)
    checkpoint = output / 'train' / f'epoch_{epoch}.pt'
    sd53._atomic_json(output / 'validation.json', {'seed': sd9.VALIDATION_SEED, 'results': summaries, 'selected_epoch': epoch})
    sd53._atomic_json(output / 'selected.json', {'epoch': epoch, 'checkpoint': str(checkpoint),
                                                 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    print(json.dumps({'selected_epoch': epoch, 'results': summaries}, indent=2), flush=True)


def test(smoke=False, workers=5):
    from run_sd78_search_test import summarize
    root = OUTPUT.parent / (OUTPUT.name + '-smoke') if smoke else OUTPUT
    sd9._require(S3_ORIGINAL, S3_SHA256)
    sd9._require(sd9.S2, sd9.S2_SHA256)
    models = {'S3_seed530': S3_ORIGINAL}
    for rep in REPS:
        _registration(root / rep, config_for(rep, smoke))
        chosen = json.loads((root / rep / 'selected.json').read_text(encoding='utf-8'))
        sd9._require(chosen['checkpoint'], chosen['sha256'])
        models[f'S3_seed{REPS[rep][2]}'] = Path(chosen['checkpoint'])
    seed, games = (sd9.VALIDATION_SEED, 4) if smoke else (TEST_SEED, sd9.GAMES)
    with open(root / 'test_opened.json', 'x', encoding='utf-8') as handle:  # the test block is used once
        json.dump({'seed': seed, 'games': games, 'models': {k: str(v) for k, v in models.items()}}, handle, indent=2)
    arms = {**{k: (v, False, ('E',)) for k, v in models.items()}, 'S2': (sd9.S2, False, ('E',))}
    results = sd9.run_jobs(arms, seed, games, sd9.MAX_STEPS, workers, f'SD9-SEEDS test seed {seed}')
    for key, rows in results.items():
        (root / f'evaluation_{key}_{seed}.json').write_text(json.dumps({'games': rows}), encoding='utf-8')
    paired = {k: sd9.paired(results[f'{k}_E'], results['S2_E']) for k in models}
    deltas = [paired[k]['active_win']['difference_pp'] for k in models]
    summary = {'seed': seed, 'games': games, 'results': {k: summarize(v) for k, v in results.items()},
               'paired_vs_S2': paired, 'gate_vs_S2': {k: sd9.gate(paired[k]) for k in models},
               'active_delta_over_seeds': {'mean': float(np.mean(deltas)), 'min': min(deltas), 'max': max(deltas),
                                           'sd': float(np.std(deltas, ddof=1))}}
    sd53._atomic_json(root / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('gate_vs_S2', 'active_delta_over_seeds')}, indent=2), flush=True)


def main(argv=None):
    import torch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('collect', 'train', 'validate', 'test'))
    p.add_argument('--rep', choices=sorted(REPS))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args(argv)
    if a.stage != 'test' and not a.rep:
        p.error('--rep is required for collect/train/validate')
    {'collect': lambda: collect(a.rep, a.smoke, a.workers, a.resume),
     'train': lambda: train(a.rep, a.smoke, a.device),
     'validate': lambda: validate(a.rep, a.smoke, a.workers),
     'test': lambda: test(a.smoke, a.workers)}[a.stage]()


if __name__ == '__main__':
    main()
