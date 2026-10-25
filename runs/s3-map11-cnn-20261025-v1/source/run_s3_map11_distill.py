"""Separate S3 -> 11x11 distillation runner. No experiment runs on import.

Initialize a fresh registered run, then explicitly call baseline, collect,
train, validate and test. See S3_MAP11_GUIDE.md for cmd commands and scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from time import monotonic

import numpy as np

from progress import bar, parallel_map
import run_sd53_pilot as sd53

ROOT = Path(__file__).resolve().parent
BOARD_SIZE, NUM_AGENTS, CHANNELS = 11, 2, 13
S3 = ROOT / 'runs/sd9-20261023/train/epoch_12.pt'
S3_SHA256 = '4185375f8b5f5cf070d3eef6ce98bec27e23fe06c64d97b500c939e36e669ed7'
SCHEDULE_E = dict(sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)
SOURCE_FILES = ('run_s3_map11_distill.py', 'bomberman_env.py', 'rule_based_bot.py',
                'train_actor_critic.py', 'evaluate.py', 'kill_search.py', 'progress.py',
                'run_sd53_pilot.py', 'benchmark_t3.py', 'pilot_two_bomb.py',
                'tactical_teacher.py', 'combat_move_teacher.py', 'training_state.py',
                'audit_l2_candidates.py', 'audit_t2.py', 'visualize.py')
# Used/reserved seeds documented in CLAUDE.md on 2026-10-24. This guard does
# not replace checking the project seed ledger before registering a new run.
USED_TEST_SEEDS = (16000, 17000, 25000, 26000, 31000, 32000, 33000, 34000,
                   35000, 36000, 37000, 38000, 39000, 40000, 41000, 42000,
                   43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000,
                   57000, 58000, 60000, 630000, 640000, 720000)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _overlap(left, right):
    return left[0] <= right[1] and right[0] <= left[1]


def _fresh_test_seed(seed, boards):
    interval = (seed, seed + boards - 1)
    occupied = [(s, s + 199) for s in USED_TEST_SEEDS] + [(51000, 56099)]
    for marker in (ROOT / 'runs').glob('*/test_opened.json'):
        opened = json.loads(marker.read_text(encoding='utf-8'))
        if isinstance(opened.get('seed'), int):
            occupied.append((opened['seed'], opened['seed'] + opened.get('games', 400) // 2 - 1))
    if any(_overlap(interval, other) for other in occupied):
        raise ValueError('test seed range is used/opened/reserved; choose a fresh test seed')


def config_for(output, *, collect_first_seed, collect_boards, validation_seed, test_seed, smoke=False):
    for name, value in (('collect_first_seed', collect_first_seed), ('collect_boards', collect_boards),
                        ('validation_seed', validation_seed), ('test_seed', test_seed)):
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if collect_boards < 2:
        raise ValueError('collect needs at least 2 board seeds')
    count, games = (4, 4) if smoke else (collect_boards, 400)
    ranges = [(collect_first_seed, collect_first_seed + count - 1),
              (validation_seed, validation_seed + games // 2 - 1),
              (test_seed, test_seed + games // 2 - 1)]
    if any(end > np.iinfo(np.uint32).max for _, end in ranges):
        raise ValueError('seed range exceeds the NumPy uint32 seed limit')
    if any(_overlap(ranges[a], ranges[b]) for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('collect/validation/test seed ranges must not overlap')
    return dict(protocol='S3 11x11 search3 distillation', output=str(Path(output).resolve()),
                board_size=BOARD_SIZE, num_agents=NUM_AGENTS, obs_channels=CHANNELS,
                s3_checkpoint=str(S3), s3_sha256=S3_SHA256, max_steps=12 if smoke else 500,
                collect_first_seed=collect_first_seed, collect_boards=count,
                collect_board_seeds=list(range(collect_first_seed, collect_first_seed + count)),
                collect_games=count * 2, collection_shard_boards=2 if smoke else 25,
                schedule_e=SCHEDULE_E.copy(), schedule_e_probability=0.75, anchor_probability=0.05,
                baseline_games=4 if smoke else 100, evaluation_games=games,
                validation_seed=validation_seed, test_seed=test_seed,
                learning_rate=1e-4, batch_size=16 if smoke else 256, train_epochs=2 if smoke else 16,
                candidate_epochs=[2] if smoke else [4, 8, 12, 16], anchor_lambda=1.0,
                heldout_fraction=0.10, smoke=bool(smoke))


def _sources():
    return {name: sd53.normalized_sha256(ROOT / name) for name in SOURCE_FILES}


def _require_s3():
    if _hash(S3) != S3_SHA256:
        raise ValueError(f'S3 checkpoint hash changed: {S3}')


def initialize(output, *, collect_first_seed, collect_boards=100, validation_seed=21000,
               test_seed, smoke=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = config_for(output, collect_first_seed=collect_first_seed, collect_boards=collect_boards,
                        validation_seed=validation_seed, test_seed=test_seed, smoke=smoke)
    if not smoke:
        _fresh_test_seed(test_seed, config['evaluation_games'] // 2)
    _require_s3()
    registration = dict(config=config, config_sha256=_digest(config), s3_sha256=S3_SHA256,
                        sources_normalized_lf_sha256=_sources(), runtime=sd53._runtime())
    output.mkdir(parents=True)
    bundle = output / 'source'; bundle.mkdir()
    for name in SOURCE_FILES:
        shutil.copyfile(ROOT / name, bundle / name)
    sd53._atomic_json(output / 'registration.json', registration)
    return registration


def _registration(output):
    output = Path(output)
    registration = json.loads((output / 'registration.json').read_text(encoding='utf-8'))
    config = registration['config']
    expected = config_for(output, collect_first_seed=config['collect_first_seed'],
                          collect_boards=config['collect_boards'], validation_seed=config['validation_seed'],
                          test_seed=config['test_seed'], smoke=config['smoke'])
    if (config != expected or registration.get('config_sha256') != _digest(config)
            or registration.get('s3_sha256') != S3_SHA256):
        raise ValueError('registration/config/checkpoint hash changed; refused')
    if registration.get('sources_normalized_lf_sha256') != _sources():
        raise ValueError('source hash differs from registration; refused')
    _require_s3()
    return config


def load_net(path, device='cpu'):
    from train_actor_critic import ActorCriticNet, smart_load_weights
    if Path(path).resolve() == S3.resolve():
        _require_s3()
    net = ActorCriticNet(in_channels=CHANNELS, board_size=BOARD_SIZE).to(device)
    smart_load_weights(net, str(path), device)
    return net.eval()


def _workers(workers):
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError('workers must be between 1 and 5')


def _absent(*paths):
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(path)


def _shards(config):
    seeds, width = config['collect_board_seeds'], config['collection_shard_boards']
    return [(seeds[i], seeds[min(i + width, len(seeds)) - 1]) for i in range(0, len(seeds), width)]


def _shard_meta(start, end, config):
    return dict(first_seed=start, last_seed=end, s3_sha256=S3_SHA256,
                config_sha256=_digest(config), sources=_sources())


def _record_game(seed, aid, config, net):
    import torch
    from bomberman_env import BombermanFFAEnv
    from kill_search import search_action3
    from rule_based_bot import RuleBasedAgent
    from train_actor_critic import get_action_mask
    np.random.seed(seed); torch.manual_seed(seed)
    use_e = sd53._schedule_e(seed)
    env = BombermanFFAEnv(board_size=BOARD_SIZE, num_agents=NUM_AGENTS, seed=seed,
                          max_steps=config['max_steps'], sudden_death=use_e, **config['schedule_e'])
    observations = env.reset()
    bots = {other: RuleBasedAgent(other) for other in range(NUM_AGENTS) if other != aid}
    decisions, anchors = [], []
    for step in range(config['max_steps']):
        obs = np.asarray(observations[aid], dtype=np.float32)
        mask = np.asarray(get_action_mask(env, aid), dtype=bool)
        sampled, logits = sd53._sample_f(net, obs, mask, 'cpu')
        action, search = search_action3(env, aid, mask.tolist(), sampled)
        scores = np.full(6, np.nan, dtype=np.float32)
        for move, score in {**search.get('scores', {}), **search.get('plans', {}),
                            **search.get('deep_plans', {})}.items():
            scores[move] = score
        base = dict(obs=obs, mask=mask, seed=seed, corner=aid, step=step)
        if not search.get('skipped', True) and len(set(scores[~np.isnan(scores)].tolist())) > 1:
            decisions.append(dict(base, scores=scores, sampled_action=sampled, executed_action=action))
        elif sd53._anchor(seed, aid, step):
            anchors.append(dict(base, logits=logits))
        actions = [0] * NUM_AGENTS
        for other, bot in bots.items():
            if env.agents[other].alive:
                actions[other] = bot.select_action(env)
        actions[aid] = action
        observations, _, _, info = env.step(actions)
        if not env.agents[aid].alive or info['episode_done']:
            break
    alive = [i for i, state in enumerate(info['alive']) if state]
    win = len(alive) == 1 and alive[0] == aid
    game = dict(seed=seed, aid=aid, win=win, active_win=bool(win and env.agents[aid].kills > 0),
                draw=len(alive) != 1, steps=env.step_count, schedule_e=use_e)
    return game, decisions, anchors


def _write_shard(path, metadata, decisions, anchors, games):
    payload = dict(metadata=np.array(json.dumps(metadata)), games=np.array(json.dumps(games)))
    for prefix, records in (('decision', decisions), ('anchor', anchors)):
        shapes = dict(obs=((CHANNELS, BOARD_SIZE, BOARD_SIZE), np.float32), mask=((6,), bool),
                      seed=((), np.int64), corner=((), np.int8), step=((), np.int16))
        shapes.update(dict(scores=((6,), np.float32), sampled_action=((), np.int8),
                           executed_action=((), np.int8)) if prefix == 'decision'
                      else dict(logits=((6,), np.float32)))
        for key, (shape, dtype) in shapes.items():
            payload[f'{prefix}_{key}'] = (np.asarray([r[key] for r in records], dtype=dtype)
                                          if records else np.empty((0, *shape), dtype=dtype))
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def _collect_shard(start, end, output, config, ticks):
    import torch
    torch.set_num_threads(1)
    net = load_net(S3)
    games, decisions, anchors = [], [], []
    for seed in range(start, end + 1):
        for aid in range(NUM_AGENTS):
            game, chosen, anchored = _record_game(seed, aid, config, net)
            games.append(game); decisions.extend(chosen); anchors.extend(anchored); ticks.put(1)
    _write_shard(Path(output) / 'collection' / f'shard-{start}-{end}.npz',
                 _shard_meta(start, end, config), decisions, anchors, games)


def _read_shard(path, start, end, config):
    try:
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        if json.loads(str(data['metadata'])) != _shard_meta(start, end, config):
            raise ValueError('metadata mismatch')
        games = json.loads(str(data['games']))
        pairs = sorted((g['seed'], g['aid']) for g in games)
        if pairs != [(s, a) for s in range(start, end + 1) for a in range(NUM_AGENTS)]:
            raise ValueError('board/corner coverage mismatch')
        for prefix, target in (('decision', 'scores'), ('anchor', 'logits')):
            count = len(data[f'{prefix}_seed'])
            for key, shape in (('obs', (count, CHANNELS, BOARD_SIZE, BOARD_SIZE)),
                               ('mask', (count, 6)), (target, (count, 6)),
                               ('seed', (count,)), ('corner', (count,)), ('step', (count,))):
                if data[f'{prefix}_{key}'].shape != shape:
                    raise ValueError(f'{prefix}_{key} shape must be {shape}')
            if not np.isfinite(data[f'{prefix}_obs']).all():
                raise ValueError('nonfinite observation')
            if not np.isin(data[f'{prefix}_seed'], list(range(start, end + 1))).all():
                raise ValueError('record seed outside shard')
            if not np.isin(data[f'{prefix}_corner'], (0, 1)).all():
                raise ValueError('invalid corner')
            if count and not data[f'{prefix}_mask'].any(axis=1).all():
                raise ValueError('record has no legal action')
        if not np.isfinite(data['anchor_logits']).all():
            raise ValueError('nonfinite anchor logits')
        return data, games
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise ValueError(f'invalid shard {path}: {error}') from error


def collect(output, workers=5, resume=False):
    _workers(workers)
    output = Path(output); config = _registration(output)
    collection = output / 'collection'
    if collection.exists() and not resume:
        raise FileExistsError(f'{collection} exists; only interrupted collect supports --resume')
    collection.mkdir(exist_ok=True)
    expected = {collection / f'shard-{s}-{e}.npz' for s, e in _shards(config)}
    if set(collection.glob('shard-*.npz')) - expected:
        raise ValueError('unexpected collection shard')
    jobs = []
    for start, end in _shards(config):
        path = collection / f'shard-{start}-{end}.npz'
        if path.exists():
            _read_shard(path, start, end, config)
        else:
            jobs.append((start, end, str(output), config))
    if jobs:
        parallel_map(_collect_shard, jobs, total=sum((e-s+1)*2 for s, e, *_ in jobs),
                     desc='S3 map11 collect', workers=min(workers, len(jobs)))
    games, decisions, anchors = [], 0, 0
    for start, end in _shards(config):
        data, rows = _read_shard(collection / f'shard-{start}-{end}.npz', start, end, config)
        games.extend(rows); decisions += len(data['decision_seed']); anchors += len(data['anchor_seed'])
    summary = dict(games=len(games), decisions=decisions, anchors=anchors, board_size=BOARD_SIZE,
                   smoke=config['smoke'], teacher_rates={k: 100*sum(g[k] for g in games)/len(games)
                                                          for k in ('win', 'active_win', 'draw')})
    path = output / 'collection_summary.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != summary:
            raise ValueError('completed collection summary differs; refused')
    else:
        sd53._atomic_json(path, summary)
    return summary


def _load_records(output, config):
    expected = {output / 'collection' / f'shard-{s}-{e}.npz' for s, e in _shards(config)}
    if set((output / 'collection').glob('shard-*.npz')) != expected:
        raise ValueError('collection incomplete or unexpected shards')
    keys = ('decision_obs', 'decision_mask', 'decision_scores', 'decision_seed',
            'anchor_obs', 'anchor_mask', 'anchor_logits', 'anchor_seed')
    chunks = {k: [] for k in keys}
    for start, end in _shards(config):
        data, _ = _read_shard(output / 'collection' / f'shard-{start}-{end}.npz', start, end, config)
        for key in keys:
            chunks[key].append(data[key])
    return {k: np.concatenate(parts) for k, parts in chunks.items()}


def train(output, device='cpu'):
    import torch
    import torch.nn.functional as functional
    output = Path(output); config = _registration(output)
    destination = output / 'train'; _absent(destination)
    records = _load_records(output, config)
    if not len(records['decision_seed']) or not len(records['anchor_seed']):
        raise ValueError('insufficient decision/anchor records; collect more boards in a new run')
    target = np.stack([sd53.masked_soft_target(s, m) for s, m in
                       zip(records['decision_scores'], records['decision_mask'])])
    held = sd53._heldout_board_ids(np.concatenate((records['decision_seed'], records['anchor_seed'])),
                                   config['heldout_fraction'], 530)
    d_train, d_held = sd53._partition_indices(records['decision_seed'], held)
    a_train, a_held = sd53._partition_indices(records['anchor_seed'], held)
    d_train = np.intersect1d(d_train, np.flatnonzero(target.sum(1) > 0))
    d_held = np.intersect1d(d_held, np.flatnonzero(target.sum(1) > 0))
    if not all(len(indices) for indices in (d_train, d_held, a_train, a_held)):
        raise ValueError('insufficient decision/anchor records in the board-level train/held-out split')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    net = load_net(S3, device).train()
    net.critic_head.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=config['learning_rate'])
    half, rng = config['batch_size'] // 2, np.random.default_rng(530)
    batches = max(-(-len(d_train)//half), -(-len(a_train)//half))
    tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)
    metrics = [dict(epoch=0, **sd53._heldout_metrics(net.eval(), records, d_held, a_held, device))]
    destination.mkdir(); net.train()
    with bar(config['train_epochs'] * batches, 'S3 map11 distill') as shown:
        for epoch in range(1, config['train_epochs'] + 1):
            order_d, order_a, losses = rng.permutation(d_train), rng.permutation(a_train), []
            for batch in range(batches):
                di = order_d[(np.arange(half) + batch*half) % len(order_d)]
                ai = order_a[(np.arange(half) + batch*half) % len(order_a)]
                logits, _ = net(tensor(records['decision_obs'][di]))
                log = functional.log_softmax(sd53._masked_logits(logits, tensor(records['decision_mask'][di], torch.bool)), 1)
                ce = -(tensor(target[di])*log).sum(1).mean()
                logits, _ = net(tensor(records['anchor_obs'][ai]))
                mask = tensor(records['anchor_mask'][ai], torch.bool)
                teacher = functional.softmax(sd53._masked_logits(tensor(records['anchor_logits'][ai]), mask), 1)
                kl = functional.kl_div(functional.log_softmax(sd53._masked_logits(logits, mask), 1),
                                       teacher, reduction='batchmean')
                loss = ce + config['anchor_lambda']*kl
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite distillation loss')
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item()); shown.update(1)
            metrics.append(dict(epoch=epoch, train_loss=float(np.mean(losses)),
                                **sd53._heldout_metrics(net.eval(), records, d_held, a_held, device)))
            net.train()
            if epoch in config['candidate_epochs']:
                torch.save(net.state_dict(), destination / f'epoch_{epoch}.pt')
    split = dict(heldout_board_seeds=sorted(held), decision_train=len(d_train), decision_heldout=len(d_held),
                 anchor_train=len(a_train), anchor_heldout=len(a_held))
    result = dict(metrics=metrics, split=split, smoke=config['smoke'], board_size=BOARD_SIZE)
    sd53._atomic_json(destination / 'metrics.json', result)
    return result


def _play(name, path, search, seed, games, max_steps, ticks):
    import torch
    import evaluate
    from kill_search import search_action3
    from train_actor_critic import get_action_mask
    torch.set_num_threads(1)
    policy = evaluate.make_ac_fn(path, BOARD_SIZE, 'cpu')
    agent = policy if not search else lambda env, obs, aid: search_action3(
        env, aid, get_action_mask(env, aid), policy(env, obs, aid))[0]
    done = [0]
    def advance(value, total):
        ticks.put(value-done[0]); done[0] = value
    result = evaluate.evaluate_seeded(agent, BOARD_SIZE, NUM_AGENTS, games, seed, max_steps,
                                      sudden_death=True, on_progress=advance, **SCHEDULE_E)
    return name, result['games']


def _run_jobs(arms, seed, games, config, workers, desc):
    jobs = [(name, str(path), search, seed, games, config['max_steps'])
            for name, (path, search) in arms.items()]
    results = parallel_map(_play, jobs, total=len(jobs)*games, desc=desc, workers=min(workers, len(jobs)))
    return {name: rows for name, rows in results}


def _summarize(games):
    return dict(num_episodes=len(games), **{f'{key}_rate': 100*sum(g[key] for g in games)/len(games)
                for key in ('win', 'active_win', 'draw', 'passive_win', 'suicide', 'sudden_death_death')})


def _paired(left, right):
    from pilot_two_bomb import paired_active_bootstrap
    return {key: paired_active_bootstrap(*({'games': [dict(g, active_win=g[key]) for g in rows]}
                                           for rows in (left, right)))
            for key in ('active_win', 'win', 'draw', 'suicide')}


def baseline(output, workers=5):
    _workers(workers)
    output = Path(output); config = _registration(output)
    path = output / 'baseline.json'; _absent(path)
    started = monotonic()
    games = _run_jobs({'S3': (S3, False), 'S3_search3': (S3, True)}, config['validation_seed'],
                      config['baseline_games'], config, workers, 'S3 map11 baseline')
    paired = _paired(games['S3_search3'], games['S3'])
    result = dict(smoke=config['smoke'], board_size=BOARD_SIZE, seed=config['validation_seed'],
                  results={name: _summarize(rows) for name, rows in games.items()}, games=games,
                  paired_teacher_vs_S3=paired, teacher_has_advantage=paired['active_win']['difference_pp'] > 0,
                  seconds=monotonic()-started)
    sd53._atomic_json(path, result)
    return result


def validate(output, workers=5):
    _workers(workers)
    output = Path(output); config = _registration(output)
    _absent(output / 'selected.json', output / 'validation.json', output / 'test_opened.json')
    arms = {'S3': (S3, False), **{f'epoch_{e}': (output / 'train' / f'epoch_{e}.pt', False)
                                      for e in config['candidate_epochs']}}
    for path, _ in arms.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    games = _run_jobs(arms, config['validation_seed'], config['evaluation_games'], config, workers,
                      'S3 map11 validation')
    summaries = {name: _summarize(rows) for name, rows in games.items()}
    limit = summaries['S3']['suicide_rate'] + 1.0
    safe = [(e, summaries[f'epoch_{e}']) for e in config['candidate_epochs']
            if summaries[f'epoch_{e}']['suicide_rate'] <= limit]
    epoch = max(safe, key=lambda item: (item[1]['active_win_rate'], -item[0]))[0] if safe else None
    improved = epoch is not None and summaries[f'epoch_{epoch}']['active_win_rate'] > summaries['S3']['active_win_rate']
    selected = dict(epoch=epoch, checkpoint=str(output / 'train' / f'epoch_{epoch}.pt') if epoch else None,
                    proceed_to_test=improved, smoke=config['smoke'])
    if epoch:
        selected['checkpoint_sha256'] = _hash(selected['checkpoint'])
    result = dict(results=summaries, selected=selected, seed=config['validation_seed'], smoke=config['smoke'])
    sd53._atomic_json(output / 'validation.json', result)
    sd53._atomic_json(output / 'selected.json', selected)
    return result


def test(output, workers=5):
    _workers(workers)
    output = Path(output); config = _registration(output)
    _absent(output / 'test_opened.json', output / 'summary.json')
    selected = json.loads((output / 'selected.json').read_text(encoding='utf-8'))
    if not selected.get('proceed_to_test'):
        raise ValueError('validation did not improve over S3 baseline; VALIDATION_STOP')
    epoch = selected['epoch']
    checkpoint = output / 'train' / f'epoch_{epoch}.pt'
    if (epoch not in config['candidate_epochs'] or Path(selected['checkpoint']).resolve() != checkpoint.resolve()
            or selected.get('checkpoint_sha256') != _hash(checkpoint)):
        raise ValueError('selected checkpoint changed; refused')
    seed = config['validation_seed'] if config['smoke'] else config['test_seed']
    if not config['smoke']:
        _fresh_test_seed(seed, config['evaluation_games']//2)
    with (output / 'test_opened.json').open('x', encoding='utf-8') as handle:
        json.dump(dict(seed=seed, games=config['evaluation_games'], smoke=config['smoke'],
                       selected=selected, s3_sha256=S3_SHA256), handle, indent=2)
    games = _run_jobs({'S3': (S3, False), 'S3_11': (checkpoint, False)}, seed,
                      config['evaluation_games'], config, workers, 'S3 map11 test')
    paired = _paired(games['S3_11'], games['S3'])
    checks = dict(active_5pp=paired['active_win']['difference_pp'] >= 5,
                   active_ci_lower_positive=paired['active_win']['ci95_pp'][0] > 0,
                   suicide_within_1pp=paired['suicide']['difference_pp'] <= 1)
    result = dict(smoke=config['smoke'], seed=seed, board_size=BOARD_SIZE, selected=selected,
                  results={name: _summarize(rows) for name, rows in games.items()}, games=games,
                  paired=paired, gate=dict(checks, **{'pass': all(checks.values())}))
    sd53._atomic_json(output / 'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest='stage', required=True)
    for name in ('init', 'baseline', 'collect', 'train', 'validate', 'test'):
        stage = stages.add_parser(name)
        stage.add_argument('--output', required=True, type=Path)
        if name == 'init':
            stage.add_argument('--collect-first-seed', type=int, required=True)
            stage.add_argument('--collect-boards', type=int, default=100)
            stage.add_argument('--validation-seed', type=int, default=21000)
            stage.add_argument('--test-seed', type=int, required=True)
            stage.add_argument('--smoke', action='store_true')
        elif name == 'train':
            stage.add_argument('--device', default='cpu')
        else:
            stage.add_argument('--workers', type=int, default=5)
            if name == 'collect':
                stage.add_argument('--resume', action='store_true')
    args = vars(parser.parse_args(argv)); stage = args.pop('stage')
    function = initialize if stage == 'init' else globals()[stage]
    result = function(**args)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
