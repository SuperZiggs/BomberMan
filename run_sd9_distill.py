"""SD9: distil S2 + three-move kill-search (`kill_search.search_action3`) into a pure CNN, starting from S2.

Copy of SD8A's runner (`run_sd8a_distill.py`, kept unchanged because its finished run checks its own source
hash) with the changes fixed before running (spec docs/superpowers/specs/2026-10-23-sd9-search3-distillation.md):
the teacher is S2+search3 (SD9.0 rule), the student starts from and is KL-anchored to S2, labels add the
three-move plan scores, new collection boards, test seed 57000.

  collect   3000 games of S2+search3 on boards 212000-213499 (schedule E with p 0.75). Decision states =
            states whose merged search scores differ (one-move scores, overridden by two-move plan scores,
            overridden by three-move plan scores); anchors = 5% of the other states with S2's logits.
  train     soft-target CE on decision states + KL to S2 on anchors (lambda 1), Adam 1e-4, batch 256,
            16 epochs, candidates 4/8/12/16; held-out = 10% of boards.
  validate  S2 and the candidates on seed 21000 (400 games, schedule E); select highest ActiveWR with
            suicide <= S2 + 1pp.
  test      seed 57000, once: selected (S3), S2 and G pure CNN on E and off; S2+search3 and G+search3 on E
            are report-only ceilings. Gate: S3 - S2 ActiveWR >= +5pp, CI95 lower > 0, suicide <= +1pp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from progress import bar, parallel_map
import run_sd53_pilot as sd53
import run_sd76_pilot as sd76

ROOT = Path(__file__).resolve().parent
G, G_SHA256 = sd76.G, sd76.G_SHA256
S2 = ROOT / 'runs/sd8a-20261020/train/epoch_16.pt'
S2_SHA256 = 'd222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814'
SCHEDULE_E = sd53.SCHEDULE_E
COLLECT_FIRST, COLLECT_LAST = 212000, 213499
VALIDATION_SEED, TEST_SEED, GAMES, MAX_STEPS = 21000, 57000, 400, 500
SOURCE_FILES = ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py', 'evaluate.py', 'kill_search.py',
                'progress.py', 'pilot_two_bomb.py', 'run_sd53_pilot.py', 'run_sd9_distill.py')


def config_for(output, smoke):
    boards = list(range(COLLECT_FIRST, COLLECT_LAST + 1))[:4 if smoke else None]
    return {'protocol': 'SD9 three-move search distillation from S2', 'output': str(Path(output)),
            'max_steps': MAX_STEPS, 's2_sha256': S2_SHA256, 'schedule_e': SCHEDULE_E,
            'collect_board_seeds': boards, 'collect_games': len(boards) * 2, 'schedule_e_probability': 0.75,
            'anchor_probability': 0.05, 'collection_shard_boards': 2 if smoke else 25,
            'train_epochs': 2 if smoke else 16, 'candidate_epochs': [2] if smoke else [4, 8, 12, 16],
            'learning_rate': 1e-4, 'batch_size': 16 if smoke else 256, 'anchor_lambda': 1.0,
            'heldout_fraction': 0.10, 'validation_seed': VALIDATION_SEED, 'test_seed': TEST_SEED,
            'evaluation_games': 4 if smoke else GAMES, 'smoke': bool(smoke)}


def _sources():
    return {name: sd53.normalized_sha256(ROOT / name) for name in SOURCE_FILES}


def _require(path, sha):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
        raise ValueError(f'checkpoint changed: {path}')


def _registration(output, config, create=False):
    marker = Path(output) / 'registration.json'
    if create:
        Path(output).mkdir(parents=True)  # FileExistsError: never overwrite a run
        registration = {'config': config, 'sources_normalized_lf_sha256': _sources(), 'runtime': sd53._runtime()}
        sd53._atomic_json(marker, registration)
        return registration
    registration = json.loads(marker.read_text(encoding='utf-8'))
    if registration['config'] != config:
        raise ValueError('configuration differs from registration; refused')
    if registration['sources_normalized_lf_sha256'] != _sources():
        raise ValueError('source hash differs from registration; refused')
    return registration


def teacher_scores(info, actions=6):
    """Per-action label scores: one-move scores, overridden by two-move then three-move plan scores where made."""
    scores = np.full(actions, np.nan, dtype=np.float32)
    for action, score in {**info.get('scores', {}), **info.get('plans', {}), **info.get('deep_plans', {})}.items():
        scores[action] = score
    return scores


def _record_game(seed, aid, config, net):
    import torch
    from bomberman_env import BombermanFFAEnv
    from kill_search import search_action3
    from rule_based_bot import RuleBasedAgent
    from train_actor_critic import get_action_mask
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_e = sd53._schedule_e(seed)
    env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=config['max_steps'], seed=seed,
                          sudden_death=use_e, **SCHEDULE_E)
    obs_list, bots = env.reset(), {other: RuleBasedAgent(other) for other in range(2) if other != aid}
    decisions, anchors, info = [], [], {'alive': [True, True]}
    for step in range(config['max_steps']):
        if not env.agents[aid].alive:
            break
        obs, mask = np.asarray(obs_list[aid], dtype=np.float32), np.asarray(get_action_mask(env, aid), dtype=bool)
        sampled, logits = sd53._sample_f(net, obs, mask, 'cpu')
        executed, search = search_action3(env, aid, mask.tolist(), sampled)
        scores = teacher_scores(search, len(mask))
        base = {'obs': obs, 'mask': mask, 'seed': seed, 'corner': aid, 'step': step}
        if not search.get('skipped', True) and len(set(scores[~np.isnan(scores)].tolist())) > 1:
            decisions.append({**base, 'scores': scores, 'sampled_action': sampled, 'executed_action': executed})
        elif sd53._anchor(seed, aid, step):
            anchors.append({**base, 'logits': logits})
        actions = [0, 0]
        for other, bot in bots.items():
            if env.agents[other].alive:
                actions[other] = bot.select_action(env)
        actions[aid] = executed
        obs_list, _, _, info = env.step(actions)
        if info['episode_done']:
            break
    alive = [i for i, state in enumerate(info['alive']) if state]
    win = len(alive) == 1 and alive[0] == aid
    game = {'seed': seed, 'aid': aid, 'win': win, 'active_win': bool(win and env.agents[aid].kills > 0),
            'draw': len(alive) != 1, 'steps': env.step_count, 'schedule_e': use_e}
    return game, decisions, anchors


def _shard_meta(start, end, config):
    return {'first_seed': start, 'last_seed': end, 'S2_sha256': S2_SHA256, 'sources': _sources(),
            'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}


def _collect_shard(start, end, output, config, ticks):
    import torch
    torch.set_num_threads(1)
    net = sd53._load_net(S2, 'cpu')
    games, decisions, anchors = [], [], []
    for seed in range(start, end + 1):
        for aid in (0, 1):
            game, found, anchored = _record_game(seed, aid, config, net)
            games.append(game); decisions.extend(found); anchors.extend(anchored)
            ticks.put(1)
    path = Path(output) / 'collection' / f'shard-{start}-{end}.npz'
    sd53._write_shard(path, _shard_meta(start, end, config), decisions, anchors, games)
    return {'games': len(games), 'decisions': len(decisions), 'anchors': len(anchors)}


def _shards(config):
    seeds, width = config['collect_board_seeds'], config['collection_shard_boards']
    return [(seeds[i], seeds[min(i + width, len(seeds)) - 1]) for i in range(0, len(seeds), width)]


def _valid_shard(path, start, end, config):
    try:
        with np.load(path, allow_pickle=False) as data:
            return json.loads(str(data['metadata'])) == _shard_meta(start, end, config)
    except (OSError, KeyError, ValueError):
        return False


def collect(output, smoke=False, workers=5, resume=False):
    _require(S2, S2_SHA256)
    output, config = Path(output), config_for(output, smoke)
    if output.exists() and not resume:
        raise FileExistsError(f'{output} exists; use --resume only for an interrupted collect')
    _registration(output, config, create=not output.exists())
    (output / 'collection').mkdir(exist_ok=True)
    jobs = []
    for start, end in _shards(config):
        path = output / 'collection' / f'shard-{start}-{end}.npz'
        if path.exists() and not _valid_shard(path, start, end, config):
            raise ValueError(f'invalid shard {path}; delete it explicitly before recollecting')
        if not path.exists():
            jobs.append((start, end, str(output), config))
    if jobs:
        parallel_map(_collect_shard, jobs, total=sum((e - s + 1) * 2 for s, e, *_ in jobs),
                     desc='SD9 collect (S2+search3)', workers=max(1, min(workers, len(jobs))))
    games, decisions, anchors = [], 0, 0
    for path in sorted((output / 'collection').glob('shard-*.npz')):
        with np.load(path, allow_pickle=False) as data:
            games += json.loads(str(data['games'])); decisions += len(data['decision_seed']); anchors += len(data['anchor_seed'])
    if len(games) != config['collect_games']:
        raise ValueError('collection is incomplete')
    summary = {'games': len(games), 'decisions': decisions, 'anchors': anchors,
               'teacher_rates': {k: 100 * sum(bool(g[k]) for g in games) / len(games) for k in ('win', 'active_win', 'draw')}}
    sd53._atomic_json(output / 'collection_summary.json', summary)
    return summary


def _load_records(output, config):
    expected = {output / 'collection' / f'shard-{s}-{e}.npz' for s, e in _shards(config)}
    if set((output / 'collection').glob('shard-*.npz')) != expected or not all(
            _valid_shard(output / 'collection' / f'shard-{s}-{e}.npz', s, e, config) for s, e in _shards(config)):
        raise ValueError('collection incomplete or invalid')
    keys = ('decision_obs', 'decision_mask', 'decision_scores', 'decision_seed',
            'anchor_obs', 'anchor_mask', 'anchor_logits', 'anchor_seed')
    chunks = {k: [] for k in keys}
    for path in sorted(expected):
        with np.load(path, allow_pickle=False) as data:
            for k in keys:
                chunks[k].append(data[k])
    return {k: np.concatenate(v) for k, v in chunks.items()}


def train(output, smoke=False, device='cpu'):
    import torch
    import torch.nn.functional as functional
    _require(S2, S2_SHA256)
    output, config = Path(output), config_for(output, smoke)
    _registration(output, config)
    records = _load_records(output, config)
    target = np.stack([sd53.masked_soft_target(s, m) for s, m in zip(records['decision_scores'], records['decision_mask'])])
    held = sd53._heldout_board_ids(np.concatenate((records['decision_seed'], records['anchor_seed'])),
                                   config['heldout_fraction'], 530)
    d_train, d_held = sd53._partition_indices(records['decision_seed'], held)
    a_train, a_held = sd53._partition_indices(records['anchor_seed'], held)
    d_train = np.intersect1d(d_train, np.flatnonzero(target.sum(1) > 0))
    if not (len(d_train) and len(a_train) and len(d_held) and len(a_held)):
        raise ValueError('insufficient records after the board-seed split')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    net = sd53._load_net(S2, device).train()
    optimizer = torch.optim.Adam([p for n, p in net.named_parameters() if not n.startswith('critic_head.')],
                                 lr=config['learning_rate'])
    rng, half = np.random.default_rng(530), config['batch_size'] // 2
    destination = output / 'train'
    destination.mkdir()  # no silent reuse
    batches = max(-(-len(d_train) // half), -(-len(a_train) // half))
    as_tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)
    metrics = [{'epoch': 0, **sd53._heldout_metrics(net.eval(), records, d_held, a_held, device)}]
    net.train()
    with bar(config['train_epochs'] * batches, 'SD9 train') as shown:
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
    sd53._atomic_json(destination / 'metrics.json', {'metrics': metrics, 'split': {
        'decision_train': int(len(d_train)), 'decision_heldout': int(len(d_held)),
        'anchor_train': int(len(a_train)), 'anchor_heldout': int(len(a_held))}})
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def play(name, path, search, rule, seed, games, max_steps, ticks):
    """One evaluation job: pure CNN (search=False) or CNN + three-move search."""
    import torch
    torch.set_num_threads(1)
    import evaluate
    import train_actor_critic as trainer
    from kill_search import search_action3
    policy = evaluate.make_ac_fn(str(path), 9)
    agent = policy if not search else (
        lambda env, obs, aid: search_action3(env, aid, trainer.get_action_mask(env, aid), policy(env, obs, aid))[0])
    done = [0]

    def advance(value, total):
        ticks.put(value - done[0]); done[0] = value
    result = evaluate.evaluate_seeded(agent, 9, 2, games, seed, max_steps, on_progress=advance,
                                      sudden_death=rule == 'E', **SCHEDULE_E)
    return name, rule, result['games']


def run_jobs(arms, seed, games, max_steps, workers, desc, shards=1):
    """arms: {name: (path, search, rules)}; search arms are split into `shards` jobs by board seed."""
    jobs = [(name, str(path), search, rule, seed + i * games // 2 // n, games // n, max_steps)
            for name, (path, search, rules) in arms.items() for rule in rules
            for n in [shards if search else 1] for i in range(n)]
    parts = parallel_map(play, jobs, total=sum(len(r) for *_, r in arms.values()) * games, desc=desc, workers=workers)
    return {f'{name}_{rule}': sorted([g for n, r, gs in parts if (n, r) == (name, rule) for g in gs],
                                     key=lambda g: (g['seed'], g['aid']))
            for name, (_, _, rules) in arms.items() for rule in rules}


def paired(left, right):
    from pilot_two_bomb import paired_active_bootstrap
    return {m: paired_active_bootstrap(*({'games': [{**g, 'active_win': g[m]} for g in rows]} for rows in (left, right)))
            for m in ('active_win', 'win', 'draw', 'passive_win', 'suicide')}


def select(summaries):
    """Highest ActiveWR among candidates with suicide <= S2 + 1pp; ties go to the earlier epoch."""
    limit = summaries['S2']['suicide_rate'] + 1.0
    safe = [(int(k.removeprefix('epoch_')), v) for k, v in summaries.items() if k != 'S2' and v['suicide_rate'] <= limit]
    if not safe:
        raise ValueError('no candidate satisfies the suicide constraint')
    return max(safe, key=lambda pair: (pair[1]['active_win_rate'], -pair[0]))[0]


def validate(output, smoke=False, workers=5):
    from run_sd78_search_test import summarize
    _require(S2, S2_SHA256)
    output, config = Path(output), config_for(output, smoke)
    _registration(output, config)
    if (output / 'selected.json').exists() or (output / 'test_opened.json').exists():
        raise FileExistsError('validation already done; selection cannot be overwritten')
    arms = {'S2': (S2, False, ('E',)), **{f'epoch_{e}': (output / 'train' / f'epoch_{e}.pt', False, ('E',))
                                       for e in config['candidate_epochs']}}
    results = run_jobs(arms, VALIDATION_SEED, config['evaluation_games'], MAX_STEPS, workers, f'SD9 validate {VALIDATION_SEED}')
    summaries = {k.removesuffix('_E'): summarize(v) for k, v in results.items()}
    epoch = select(summaries)
    summary = {'seed': VALIDATION_SEED, 'results': summaries,
               'paired_vs_S2': {k.removesuffix('_E'): paired(v, results['S2_E']) for k, v in results.items() if k != 'S2_E'},
               'selected_epoch': epoch, 'selected_checkpoint': str(output / 'train' / f'epoch_{epoch}.pt')}
    sd53._atomic_json(output / 'validation.json', summary)
    sd53._atomic_json(output / 'selected.json', {'epoch': epoch, 'checkpoint': summary['selected_checkpoint']})
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def gate(vs_s2):
    checks = {'active_vs_S2_5pp': vs_s2['active_win']['difference_pp'] >= 5.0,
              'active_vs_S2_ci_lower_positive': vs_s2['active_win']['ci95_pp'][0] > 0,
              'suicide_within_1pp': vs_s2['suicide']['difference_pp'] <= 1.0}
    return {**checks, 'pass': all(checks.values())}


def test(output, smoke=False, workers=5):
    from run_sd78_search_test import summarize
    output, config = Path(output), config_for(output, smoke)
    _registration(output, config)
    _require(G, G_SHA256)
    _require(S2, S2_SHA256)
    selected = Path(json.loads((output / 'selected.json').read_text(encoding='utf-8'))['checkpoint'])
    seed = VALIDATION_SEED if smoke else TEST_SEED
    with open(output / 'test_opened.json', 'x', encoding='utf-8') as handle:  # the test seed is used once
        json.dump({'seed': seed, 'games': config['evaluation_games'], 'models': {
            'S3': hashlib.sha256(selected.read_bytes()).hexdigest(), 'S2': S2_SHA256, 'G': G_SHA256}}, handle, indent=2)
    arms = {'S3': (selected, False, ('E', 'off')), 'S2': (S2, False, ('E', 'off')), 'G': (G, False, ('E', 'off')),
            'S2_search3': (S2, True, ('E',)), 'G_search3': (G, True, ('E',))}
    results = run_jobs(arms, seed, config['evaluation_games'], MAX_STEPS, workers, f'SD9 test seed {seed}',
                       shards=2 if smoke else 4)
    for key, rows in results.items():
        (output / f'evaluation_{key}_{seed}.json').write_text(json.dumps({'games': rows}), encoding='utf-8')
    comparisons = {f'S3_vs_{other}_{rule}': paired(results[f'S3_{rule}'], results[f'{other}_{rule}'])
                   for other in ('S2', 'G') for rule in ('E', 'off')}
    comparisons.update({f'S3_vs_{o}_E': paired(results['S3_E'], results[f'{o}_E']) for o in ('S2_search3', 'G_search3')})
    summary = {'seed': seed, 'games': config['evaluation_games'], 'selected': str(selected),
               'results': {k: summarize(v) for k, v in results.items()}, 'paired': comparisons,
               'gate': gate(comparisons['S3_vs_S2_E'])}
    sd53._atomic_json(output / 'summary.json', summary)
    if not smoke:
        shutil.make_archive(str(output) + '-send', 'zip', output, '.')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    import torch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('collect', 'train', 'validate', 'test'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true', help='collect only: continue an interrupted collection')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=5)
    a = p.parse_args(argv)
    {'collect': lambda: print(json.dumps(collect(a.output, a.smoke, a.workers, a.resume), indent=2), flush=True),
     'train': lambda: train(a.output, a.smoke, a.device),
     'validate': lambda: validate(a.output, a.smoke, a.workers),
     'test': lambda: test(a.output, a.smoke, a.workers)}[a.stage]()


if __name__ == '__main__':
    main()
