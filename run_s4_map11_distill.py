"""S4 map-11 experiment: opponent-aware distillation and guarded evaluation."""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
from progress import parallel_map
import run_sd53_pilot as sd53

ROOT = Path(__file__).resolve().parent
LEDGER_ROOTS = (ROOT, Path('E:/Bomberman'))
CHECKPOINTS = {
    'S4': ROOT / 'runs/sd10-20261026/train/epoch_12.pt',
    'S2': ROOT / 'runs/sd8a-20261020/train/epoch_16.pt',
    'G': ROOT / 'runs/sd75-20261018/full/best_phaseC.pt',
    'S3_11': ROOT / 'runs/s3-map11-20261024-auto-v1/train/epoch_16.pt',
    'S3': ROOT / 'runs/sd9-20261023/train/epoch_12.pt',
}
CHECKPOINT_HASHES = {
    'S4': '90d1d0e70c64f879e07f85c68c003969d082043b3cd8065314a3c35faf62bece',
    'S2': 'd222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814',
    'G': 'bb0c775c58161aa1bbdd229b9528170aab174d5aabf8f4a6739b848919cdc7ac',
    'S3_11': '17748471874686388f1a9b373b0debdfe53ba04a9b1c821a9695efdd8825a95f',
    'S3': '4185375f8b5f5cf070d3eef6ce98bec27e23fe06c64d97b500c939e36e669ed7',
}
CNN_OPPONENTS = ('S2', 'G', 'S3_11')
SCHEDULE_E = {'sudden_death_start': 475, 'sudden_death_interval': 25,
              'sudden_death_warning_steps': 40}


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    return value


def config_for(output, *, smoke=False, collect_first_seed=860000, pilot_seed=850000,
               validation_seed=870000, test_seed=880000):
    for value in (collect_first_seed, pilot_seed, validation_seed, test_seed):
        _seed(value)
    if smoke:
        collect_first_seed, pilot_seed, validation_seed, test_seed = 900000, 900000, 900002, 900003
    collect_count, pilot_boards, eval_boards = (2, 2, 1) if smoke else (999, 10, 200)
    ranges = [(collect_first_seed, collect_first_seed + collect_count),
              (pilot_seed, pilot_seed + pilot_boards),
              (validation_seed, validation_seed + eval_boards), (test_seed, test_seed + eval_boards)]
    if any(end > 2**32 for _, end in ranges):
        raise ValueError('seed range overflows uint32')
    if not smoke and any(a < d and c < b for i, (a, b) in enumerate(ranges) for c, d in ranges[i+1:]):
        raise ValueError('seed ranges overlap')
    return {'protocol': 'S4 map11 opponent-aware distillation', 'output': str(Path(output)),
            'board_size': 11, 'num_agents': 2, 'obs_channels': 13,
            'max_steps': 12 if smoke else 500, 'collect_games': collect_count * 2,
            'collect_board_seeds': list(range(collect_first_seed, collect_first_seed + collect_count)),
            'collection_shard_boards': 2 if smoke else 20,
            'pilot_seed': pilot_seed, 'pilot_boards': pilot_boards, 'pilot_games': pilot_boards * 16,
            'validation_seed': validation_seed, 'test_seed': test_seed, 'evaluation_boards': eval_boards,
            'validation_games': eval_boards * 2 * (2 if smoke else 5) * 4,
            'test_games': eval_boards * 2 * 3 * 6,
            'candidate_epochs': [2] if smoke else [4, 8, 12, 16], 'train_epochs': 2 if smoke else 16,
            'learning_rate': 1e-4, 'batch_size': 16 if smoke else 256, 'anchor_lambda': 1.0,
            'heldout_fraction': .1, 'schedule_e': dict(SCHEDULE_E), 'schedule_e_probability': .75,
            'smoke': bool(smoke)}


def validate_workers(workers):
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError('workers must be an integer between 1 and 5')
    return workers


def opponent_for(seed, config):
    return CNN_OPPONENTS[(seed - config['collect_board_seeds'][0]) % 3]


def rates(summaries, model):
    keys = ('active_win_rate', 'suicide_rate', 'win_rate', 'draw_rate')
    return {'macro_cnn': {k: float(np.mean([summaries[f'{model}_vs_{o}'][k] for o in CNN_OPPONENTS])) for k in keys},
            'rulebot': summaries[f'{model}_vs_rulebot']}


def select(summaries, epochs=(4, 8, 12, 16)):
    base, checks = rates(summaries, 'S4'), {}
    for epoch in epochs:
        r = rates(summaries, f'epoch_{epoch}')
        checks[epoch] = {'rulebot_active_within_3pp': r['rulebot']['active_win_rate'] >= base['rulebot']['active_win_rate'] - 3,
                         'macro_suicide_within_1pp': r['macro_cnn']['suicide_rate'] <= base['macro_cnn']['suicide_rate'] + 1,
                         'rulebot_suicide_within_1pp': r['rulebot']['suicide_rate'] <= base['rulebot']['suicide_rate'] + 1,
                         'macro_active': r['macro_cnn']['active_win_rate']}
    eligible = [e for e in epochs if all(v for k, v in checks[e].items() if k != 'macro_active')]
    if not eligible:
        return None, False, checks
    best = max(eligible, key=lambda e: (checks[e]['macro_active'], -e))
    return best, checks[best]['macro_active'] > base['macro_cnn']['active_win_rate'], checks


def gate(comparison):
    macro, rule = comparison['macro_cnn'], comparison['rulebot']
    checks = {'macro_active_gain_5pp': macro['active_win']['difference_pp'] >= 5,
              'macro_ci_positive': macro['active_win']['ci95_pp'][0] > 0,
              'macro_suicide_within_1pp': macro['suicide']['difference_pp'] <= 1,
              'rulebot_active_within_3pp': rule['active_win']['difference_pp'] >= -3,
              'rulebot_suicide_within_1pp': rule['suicide']['difference_pp'] <= 1}
    return {'pass': all(checks.values()), 'checks': checks}


def pilot_ready(report):
    try:
        return (all(report['decisions_by_opponent'][o] > 0 for o in CNN_OPPONENTS)
                and report['anchors'] > 0 and np.isfinite(report['macro_active_ci95_pp']).all()
                and report['macro_active_ci95_pp'][1] >= 0)
    except (KeyError, TypeError, ValueError):
        return False


def valid_scores(scores, mask):
    scores, mask = np.asarray(scores), np.asarray(mask, dtype=bool)
    return (scores.shape == (6,) and mask.shape == (6,) and not np.isinf(scores).any()
            and np.isfinite(scores[mask]).any() and not np.isfinite(scores[~mask]).any())


def _sources():
    # Root Python modules cover the local transitive imports of the experiment.
    paths = sorted(ROOT.glob('*.py'))
    launcher = ROOT / 'run_s4_map11_pipeline.ps1'
    if launcher.is_file():
        paths.append(launcher)
    return {p.name: sd53.normalized_sha256(p) for p in paths}


def _check_checkpoints():
    for name, path in CHECKPOINTS.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_HASHES[name]:
            raise ValueError(f'checkpoint missing or hash mismatch: {name}: {path}')


def _check_seed_ledger(config):
    proposed = set(config['collect_board_seeds'])
    for key, count in (('pilot_seed', config['pilot_boards']), ('validation_seed', config['evaluation_boards']),
                       ('test_seed', config['evaluation_boards'])):
        proposed.update(range(config[key], config[key] + count))
    seen = set()
    for root in LEDGER_ROOTS:
        root = Path(root)
        for base in (root, root / 'runs'):
            if not base.is_dir():
                continue
            for pattern in ('*/test_opened.json', '*/registration.json'):
                for path in base.glob(pattern):
                    if path.resolve() in seen:
                        continue
                    seen.add(path.resolve())
                    try:
                        data = json.loads(path.read_text(encoding='utf-8'))
                        cfg = data.get('config', data)
                        occupied = set(cfg.get('collect_board_seeds', []))
                        for key, count in (('seed', data.get('boards', (data.get('games', 400) + 1) // 2)),
                                           ('validation_seed', cfg.get('evaluation_boards', (cfg.get('evaluation_games', 400) + 1) // 2)),
                                           ('test_seed', cfg.get('evaluation_boards', (cfg.get('evaluation_games', 400) + 1) // 2)),
                                           ('pilot_seed', cfg.get('pilot_boards', 1))):
                            value = (data if key == 'seed' else cfg).get(key)
                            if type(value) is int and type(count) is int and 0 < count <= 100000:
                                occupied.update(range(value, value + count))
                        if proposed.intersection(occupied):
                            raise ValueError(f'seed overlap with {path}')
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        raise ValueError(f'unreadable seed ledger: {path}') from error


def initialize(output, *, smoke=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = config_for(output, smoke=smoke)
    if not smoke:
        _check_seed_ledger(config)
    _check_checkpoints()
    sources = _sources()
    output.mkdir(parents=True)
    source = output / 'source'
    source.mkdir()
    for name in sources:
        shutil.copyfile(ROOT / name, source / name)
    registration = {'config': config, 'sources_normalized_lf_sha256': sources,
                    'checkpoints_sha256': CHECKPOINT_HASHES, 'runtime': sd53._runtime()}
    sd53._atomic_json(output / 'registration.json', registration)
    return registration


def _registration(output):
    output = Path(output)
    registration = json.loads((output / 'registration.json').read_text(encoding='utf-8'))
    config = registration['config']
    if config != config_for(output, smoke=config['smoke']):
        raise ValueError('registered config differs')
    hashes = registration['sources_normalized_lf_sha256']
    if hashes != _sources() or any(sd53.normalized_sha256(output / 'source' / name) != value
                                  for name, value in hashes.items()):
        raise ValueError('source bundle hash mismatch')
    if registration['checkpoints_sha256'] != CHECKPOINT_HASHES:
        raise ValueError('checkpoint registration changed')
    _check_checkpoints()
    return config


@contextlib.contextmanager
def run_lock(output, config_hash):
    marker = Path(output) / 'run.lock.json'
    token = uuid.uuid4().hex
    with marker.open('x', encoding='utf-8') as handle:
        json.dump({'token': token, 'config_sha256': config_hash}, handle)
    try:
        yield
    finally:
        try:
            if json.loads(marker.read_text(encoding='utf-8')).get('token') == token:
                marker.unlink()
        except (OSError, ValueError):
            pass


def load_net(path, device='cpu'):
    from train_actor_critic import ActorCriticNet, smart_load_weights
    net = ActorCriticNet(in_channels=13, board_size=11).to(device)
    smart_load_weights(net, str(path), device)
    return net.eval()


def load_opponent_models(names):
    import torch
    import evaluate
    from train_actor_critic import get_action_mask, obs_to_tensor
    models = {}
    for name in names:
        net = load_net(CHECKPOINTS[name])
        def argmax(env, aid, net=net):
            with torch.no_grad():
                logits, _ = net(obs_to_tensor(env._get_obs(aid), 'cpu'))
                mask = torch.tensor(get_action_mask(env, aid), dtype=torch.bool)
                return int(logits.reshape(-1).masked_fill(~mask, -1e9).argmax())
        models[name] = (evaluate.make_ac_fn(str(CHECKPOINTS[name]), 11), argmax)
    return models


def record_game(seed, aid, opponent, config, net, models, use_search=True):
    import time
    import torch
    import kill_search
    from bomberman_env import BombermanFFAEnv
    from rule_based_bot import RuleBasedAgent
    from train_actor_critic import get_action_mask
    from run_sd9_distill import teacher_scores
    started = time.perf_counter()
    np.random.seed(seed); torch.manual_seed(seed)
    env = BombermanFFAEnv(board_size=11, num_agents=2, max_steps=config['max_steps'], seed=seed,
                          sudden_death=sd53._schedule_e(seed), **SCHEDULE_E)
    obs_list = env.reset()
    other = 1 - aid
    bot = RuleBasedAgent(other) if opponent == 'rulebot' else None
    previous = kill_search.OPPONENT_MODEL
    kill_search.OPPONENT_MODEL = None if bot else models[opponent][1]
    decisions, anchors, info = [], [], {'alive': [True, True]}
    try:
        for step in range(config['max_steps']):
            if not env.agents[aid].alive:
                break
            obs = np.asarray(obs_list[aid], dtype=np.float32)
            mask = np.asarray(get_action_mask(env, aid), dtype=bool)
            sampled, logits = sd53._sample_f(net, obs, mask, 'cpu')
            if not np.isfinite(obs).all() or not np.isfinite(logits).all() or not mask.any() or not mask[sampled]:
                raise ValueError('invalid observation, logits, mask or sampled action')
            action = sampled
            if use_search:
                action, search = kill_search.search_action3(env, aid, mask.tolist(), sampled)
                scores = teacher_scores(search, len(mask))
                base = dict(obs=obs, mask=mask, seed=seed, corner=aid, step=step)
                if not search.get('skipped', True) and len(set(scores[~np.isnan(scores)].tolist())) > 1:
                    if not valid_scores(scores, mask):
                        raise ValueError('invalid search scores')
                    decisions.append(dict(base, scores=scores, sampled_action=sampled, executed_action=action))
                elif sd53._anchor(seed, aid, step):
                    anchors.append(dict(base, logits=logits))
            actions = [0, 0]
            if env.agents[other].alive:
                actions[other] = bot.select_action(env) if bot else models[opponent][0](env, obs_list, other)
            actions[aid] = action
            obs_list, _, _, info = env.step(actions)
            if info['episode_done']:
                break
    finally:
        kill_search.OPPONENT_MODEL = previous
    alive = [i for i, status in enumerate(info['alive']) if status]
    win = len(alive) == 1 and alive[0] == aid
    game = {'seed': seed, 'aid': aid, 'opponent': opponent, 'win': win,
            'active_win': bool(win and env.agents[aid].kills > 0), 'draw': len(alive) != 1,
            'steps': env.step_count, 'decisions': len(decisions), 'anchors': len(anchors),
            'elapsed_seconds': time.perf_counter() - started}
    return game, decisions, anchors


def _digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _pilot_job(seed, opponent, config, ticks):
    import torch
    torch.set_num_threads(1)
    net = load_net(CHECKPOINTS['S4'])
    models = load_opponent_models(()) if opponent == 'rulebot' else load_opponent_models((opponent,))
    rows = []
    for aid in (0, 1):
        for search in (False, True):
            game, decisions, anchors = record_game(seed, aid, opponent, config, net, models, use_search=search)
            game['search'] = search
            game['decisions'], game['anchors'] = len(decisions), len(anchors)
            rows.append(game)
            ticks.put(1)
    return rows


def _pilot_macro(rows, seeds):
    pairs = {(r['seed'], r['aid'], r['opponent'], r['search']): r for r in rows}
    expected = {(s, a, o, search) for s in seeds for a in (0, 1)
                for o in (*CNN_OPPONENTS, 'rulebot') for search in (False, True)}
    if set(pairs) != expected or len(rows) != len(expected):
        raise ValueError('pilot board/corner/opponent coverage mismatch')
    differences = np.array([np.mean([float(pairs[s, a, o, True]['active_win']) -
                                      float(pairs[s, a, o, False]['active_win'])
                                      for a in (0, 1) for o in CNN_OPPONENTS]) for s in seeds])
    rng = np.random.default_rng(20261014)
    draws = differences[rng.integers(0, len(seeds), size=(10000, len(seeds)))].mean(axis=1) * 100
    return float(differences.mean() * 100), [float(x) for x in np.quantile(draws, [.025, .975])]


def pilot(output, workers=5):
    validate_workers(workers)
    output = Path(output)
    config = _registration(output)
    with run_lock(output, _digest(config)):
        if (output / 'pilot.json').exists() or (output / 'pilot_opened.json').exists():
            raise FileExistsError('pilot already opened')
        sd53._atomic_json(output / 'pilot_opened.json', {'seed': config['pilot_seed'], 'boards': config['pilot_boards']})
        seeds = range(config['pilot_seed'], config['pilot_seed'] + config['pilot_boards'])
        jobs = [(s, o, config) for s in seeds for o in (*CNN_OPPONENTS, 'rulebot')]
        parts = parallel_map(_pilot_job, jobs, total=config['pilot_games'], desc='S4-11 pilot', workers=workers)
        rows = [r for part in parts for r in part]
        difference, ci = _pilot_macro(rows, seeds)
        search_rows = [r for r in rows if r['search']]
        decisions = {o: sum(r['decisions'] for r in search_rows if r['opponent'] == o) for o in CNN_OPPONENTS}
        anchors = sum(r['anchors'] for r in search_rows)
        timing = {o: float(np.mean([r['elapsed_seconds'] for r in search_rows if r['opponent'] == o]))
                  for o in (*CNN_OPPONENTS, 'rulebot')}
        report = {'games': rows, 'decisions_by_opponent': decisions, 'anchors': anchors,
                  'macro_active_difference_pp': difference, 'macro_active_ci95_pp': ci,
                  'search_seconds_per_game': timing,
                  'estimated_collect_seconds_ideal': sum(timing[o] * 666 for o in CNN_OPPONENTS) / workers}
        report['proceed_to_collect'] = pilot_ready(report)
        sd53._atomic_json(output / 'pilot.json', report)
        return report


def _shards(config):
    seeds, width = config['collect_board_seeds'], config['collection_shard_boards']
    return [(seeds[i], seeds[min(i + width, len(seeds)) - 1]) for i in range(0, len(seeds), width)]


def _shard_meta(start, end, config):
    return {'first_seed': start, 'last_seed': end, 'config_sha256': _digest(config),
            'checkpoint_hashes': CHECKPOINT_HASHES, 'sources': _sources()}


def _collect_job(start, end, output, config, ticks):
    import torch
    from run_s3_map11_distill import _write_shard
    torch.set_num_threads(1)
    net, models = load_net(CHECKPOINTS['S4']), load_opponent_models(CNN_OPPONENTS)
    games, decisions, anchors = [], [], []
    for seed in range(start, end + 1):
        for aid in (0, 1):
            game, found, anchored = record_game(seed, aid, opponent_for(seed, config), config, net, models)
            games.append(game); decisions.extend(found); anchors.extend(anchored); ticks.put(1)
    _write_shard(Path(output) / 'collection' / f'shard-{start}-{end}.npz',
                 _shard_meta(start, end, config), decisions, anchors, games)


def _read_shard(path, start, end, config):
    try:
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        if json.loads(str(data['metadata'])) != _shard_meta(start, end, config):
            raise ValueError('metadata mismatch')
        games = json.loads(str(data['games']))
        if sorted((g['seed'], g['aid']) for g in games) != [(s, a) for s in range(start, end + 1) for a in (0, 1)]:
            raise ValueError('board/corner coverage mismatch')
        if any(g['opponent'] != opponent_for(g['seed'], config) for g in games):
            raise ValueError('opponent mismatch')
        for prefix, target in (('decision', 'scores'), ('anchor', 'logits')):
            count = len(data[f'{prefix}_seed'])
            for key, shape in (('obs', (count, 13, 11, 11)), ('mask', (count, 6)),
                               (target, (count, 6)), ('seed', (count,)), ('corner', (count,)), ('step', (count,))):
                if data[f'{prefix}_{key}'].shape != shape:
                    raise ValueError(f'{prefix}_{key} shape mismatch')
            if (not np.isfinite(data[f'{prefix}_obs']).all() or
                not np.isin(data[f'{prefix}_seed'], range(start, end + 1)).all() or
                not np.isin(data[f'{prefix}_corner'], (0, 1)).all() or
                not ((0 <= data[f'{prefix}_step']) & (data[f'{prefix}_step'] < config['max_steps'])).all() or
                not data[f'{prefix}_mask'].any(axis=1).all()):
                raise ValueError('invalid observation, seed, corner, step or mask')
        if not np.isfinite(data['anchor_logits']).all():
            raise ValueError('invalid anchor logits')
        if not all(valid_scores(s, m) for s, m in zip(data['decision_scores'], data['decision_mask'])):
            raise ValueError('invalid decision scores')
        for key in ('sampled_action', 'executed_action'):
            actions = data[f'decision_{key}']
            if actions.shape != (len(data['decision_seed']),) or not ((0 <= actions) & (actions < 6)).all():
                raise ValueError('invalid decision action')
        return data, games
    except (OSError, KeyError, ValueError, EOFError) as error:
        raise ValueError(f'invalid shard {path}: {error}') from error


def collect(output, workers=5, resume=False):
    validate_workers(workers)
    output = Path(output); config = _registration(output)
    with run_lock(output, _digest(config)):
        pilot_report = json.loads((output / 'pilot.json').read_text(encoding='utf-8'))
        seeds = range(config['pilot_seed'], config['pilot_seed'] + config['pilot_boards'])
        rows = pilot_report['games']
        difference, ci = _pilot_macro(rows, seeds)
        search_rows = [r for r in rows if r['search']]
        decisions = {o: sum(r['decisions'] for r in search_rows if r['opponent'] == o) for o in CNN_OPPONENTS}
        anchors = sum(r['anchors'] for r in search_rows)
        if (decisions != pilot_report['decisions_by_opponent'] or anchors != pilot_report['anchors'] or
                difference != pilot_report['macro_active_difference_pp'] or ci != pilot_report['macro_active_ci95_pp'] or
                not pilot_ready(pilot_report) or not pilot_report.get('proceed_to_collect')):
            raise ValueError('pilot did not authorize collection')
        folder = output / 'collection'
        if folder.exists() and not resume:
            raise FileExistsError('collection already exists; resume required')
        folder.mkdir(exist_ok=True)
        expected = {folder / f'shard-{a}-{b}.npz' for a, b in _shards(config)}
        if set(folder.glob('shard-*.npz')) - expected:
            raise ValueError('unexpected shard')
        jobs = []
        for start, end in _shards(config):
            path = folder / f'shard-{start}-{end}.npz'
            if path.exists():
                _read_shard(path, start, end, config)
            else:
                jobs.append((start, end, str(output), config))
        if jobs:
            parallel_map(_collect_job, jobs, total=sum((b-a+1)*2 for a, b, *_ in jobs),
                         desc='S4-11 collection', workers=min(workers, len(jobs)))
        data = [_read_shard(folder / f'shard-{a}-{b}.npz', a, b, config) for a, b in _shards(config)]
        rows = [g for _, games in data for g in games]
        summary = {'games': len(rows), 'decisions': sum(len(d['decision_seed']) for d, _ in data),
                   'anchors': sum(len(d['anchor_seed']) for d, _ in data),
                   'by_opponent': {o: sum(g['opponent'] == o for g in rows) for o in CNN_OPPONENTS}}
        sd53._atomic_json(output / 'collection_summary.json', summary)
        return summary


def _load_records(output, config):
    folder = Path(output) / 'collection'
    expected = {folder / f'shard-{a}-{b}.npz' for a, b in _shards(config)}
    if set(folder.glob('shard-*.npz')) != expected:
        raise ValueError('collection incomplete or unexpected shard')
    keys = ('decision_obs', 'decision_mask', 'decision_scores', 'decision_seed',
            'anchor_obs', 'anchor_mask', 'anchor_logits', 'anchor_seed')
    chunks = {k: [] for k in keys}
    for start, end in _shards(config):
        data, _ = _read_shard(folder / f'shard-{start}-{end}.npz', start, end, config)
        for key in keys:
            chunks[key].append(data[key])
    return {k: np.concatenate(v) for k, v in chunks.items()}


def train(output, device='cpu'):
    import torch
    import torch.nn.functional as functional
    from progress import bar
    output = Path(output); config = _registration(output)
    with run_lock(output, _digest(config)):
        destination = output / 'train'
        if destination.exists():
            raise FileExistsError('train output already exists')
        records = _load_records(output, config)
        if not len(records['decision_seed']) or not len(records['anchor_seed']):
            raise ValueError('insufficient training records')
        target = np.stack([sd53.masked_soft_target(s, m) for s, m in zip(records['decision_scores'], records['decision_mask'])])
        held = sd53._heldout_board_ids(np.concatenate((records['decision_seed'], records['anchor_seed'])),
                                       config['heldout_fraction'], 1010)
        d_train, d_held = sd53._partition_indices(records['decision_seed'], held)
        a_train, a_held = sd53._partition_indices(records['anchor_seed'], held)
        informative = np.flatnonzero(target.sum(1) > 0)
        d_train, d_held = np.intersect1d(d_train, informative), np.intersect1d(d_held, informative)
        if not all(len(v) for v in (d_train, d_held, a_train, a_held)):
            raise ValueError('insufficient records after board-level split')
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable')
        torch.manual_seed(1010)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1010)
        net = load_net(CHECKPOINTS['S4'], device).train()
        net.critic_head.requires_grad_(False)
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=config['learning_rate'])
        rng, half = np.random.default_rng(1010), config['batch_size'] // 2
        batches = max(-(-len(d_train)//half), -(-len(a_train)//half))
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)
        metrics = [dict(epoch=0, **sd53._heldout_metrics(net.eval(), records, d_held, a_held, device))]
        destination.mkdir(); net.train()
        with bar(config['train_epochs'] * batches, 'S4-11 distillation') as shown:
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
                    kl = functional.kl_div(functional.log_softmax(sd53._masked_logits(logits, mask), 1), teacher,
                                           reduction='batchmean')
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
        hashes = {str(e): hashlib.sha256((destination / f'epoch_{e}.pt').read_bytes()).hexdigest()
                  for e in config['candidate_epochs']}
        result = {'metrics': metrics, 'checkpoint_sha256': hashes, 'device': device, 'rng_seed': 1010,
                  'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                  'heldout_board_seeds': sorted(held), 'decision_train': len(d_train), 'decision_heldout': len(d_held),
                  'anchor_train': len(a_train), 'anchor_heldout': len(a_held)}
        sd53._atomic_json(destination / 'metrics.json', result)
        return result


def _eval_job(name, path, opponent, seed, boards, config, folder, ticks):
    import torch
    import evaluate
    torch.set_num_threads(1)
    model = evaluate.make_ac_fn(path, 11, 'cpu')
    if opponent == 'rulebot':
        seat = None
    else:
        opposing = evaluate.make_ac_fn(str(CHECKPOINTS[opponent]), 11, 'cpu')
        seat = lambda aid: opposing
    count = [0]
    def advance(value, total):
        ticks.put(value - count[0]); count[0] = value
    result = evaluate.evaluate_seeded(model, 11, 2, boards * 2, seed, config['max_steps'],
                                      on_progress=advance, sudden_death=True, opponent_fn=seat, **SCHEDULE_E)
    rows = result['games']
    part = Path(folder) / f'{name}_vs_{opponent}_{seed}.json'
    sd53._atomic_json(part, {'model': name, 'opponent': opponent, 'games': rows})
    return name, opponent, rows


def _evaluation(output, stage, models, opponents, seed, config, workers):
    folder = Path(output) / f'{stage}_parts'
    folder.mkdir()
    width = 1 if config['smoke'] else 20
    boards = config['evaluation_boards']
    jobs = [(name, str(path), o, seed + i, min(width, boards-i), config, str(folder))
            for name, path in models.items() for o in opponents for i in range(0, boards, width)]
    parts = parallel_map(_eval_job, jobs, total=len(models)*len(opponents)*boards*2,
                         desc=f'S4-11 {stage}', workers=workers)
    results = {}
    for name, opponent, rows in parts:
        results.setdefault(f'{name}_vs_{opponent}', []).extend(rows)
    expected = [(s, a) for s in range(seed, seed + boards) for a in (0, 1)]
    for key, rows in results.items():
        if sorted((r['seed'], r['aid']) for r in rows) != expected:
            raise ValueError(f'evaluation coverage mismatch: {key}')
    if len(results) != len(models)*len(opponents):
        raise ValueError('missing evaluation arms')
    return results


def _summaries(results, max_steps=500):
    keys = ('win', 'active_win', 'draw', 'suicide', 'passive_win')
    return {key: {**{f'{metric}_rate': float(np.mean([g[metric] for g in games]) * 100) for metric in keys},
                  'avg_steps': float(np.mean([g['steps'] for g in games])),
                  'timeout_rate': float(np.mean([g['steps'] >= max_steps for g in games]) * 100)}
            for key, games in results.items()}


def _paired(results, left, right, opponents, metric):
    maps = {}
    for name in (left, right):
        for opponent in opponents:
            rows = results[f'{name}_vs_{opponent}']
            maps[name, opponent] = {(g['seed'], g['aid']): float(g[metric]) for g in rows}
    reference = set(maps[left, opponents[0]])
    if not reference or any(set(rows) != reference for rows in maps.values()):
        raise ValueError('paired board/corner mismatch')
    seeds = sorted({seed for seed, aid in reference})
    if reference != {(s, a) for s in seeds for a in (0, 1)}:
        raise ValueError('paired corners missing')
    differences = np.array([np.mean([maps[left, o][s, a] - maps[right, o][s, a]
                                     for o in opponents for a in (0, 1)]) for s in seeds])
    rng = np.random.default_rng(20261014)
    draws = differences[rng.integers(0, len(seeds), size=(10000, len(seeds)))].mean(axis=1) * 100
    return {'difference_pp': float(differences.mean()*100),
            'ci95_pp': [float(x) for x in np.quantile(draws, [.025, .975])],
            'board_pairs': len(seeds), 'resamples': 10000, 'bootstrap_seed': 20261014}


def _compare(results, left, right, opponents):
    metrics = ('active_win', 'win', 'draw', 'suicide', 'passive_win')
    groups = {'macro_cnn': CNN_OPPONENTS, **{o: (o,) for o in opponents}}
    return {name: {m: _paired(results, left, right, group, m) for m in metrics}
            for name, group in groups.items()}


def _candidate_paths(output, config):
    output = Path(output)
    metrics = json.loads((output / 'train' / 'metrics.json').read_text(encoding='utf-8'))
    models = {}
    for epoch in config['candidate_epochs']:
        path = output / 'train' / f'epoch_{epoch}.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest() != metrics['checkpoint_sha256'][str(epoch)]:
            raise ValueError(f'candidate checkpoint hash mismatch: {epoch}')
        models[f'epoch_{epoch}'] = path
    return models


def validate(output, workers=5):
    validate_workers(workers)
    output = Path(output); config = _registration(output)
    with run_lock(output, _digest(config)):
        if any((output / name).exists() for name in ('selected.json', 'validation.json', 'test_opened.json', 'validation_parts')):
            raise FileExistsError('validation already started or completed')
        models = {'S4': CHECKPOINTS['S4'], **_candidate_paths(output, config)}
        opponents = (*CNN_OPPONENTS, 'rulebot')
        results = _evaluation(output, 'validation', models, opponents, config['validation_seed'], config, workers)
        summaries = _summaries(results, config['max_steps'])
        epoch, proceed, checks = select(summaries, config['candidate_epochs'])
        summary = {'seed': config['validation_seed'], 'results': summaries,
                   'rates': {m: rates(summaries, m) for m in models}, 'checks': checks,
                   'selected_epoch': epoch, 'proceed_to_test': proceed}
        sd53._atomic_json(output / 'validation.json', summary)
        path = None if epoch is None else models[f'epoch_{epoch}']
        sd53._atomic_json(output / 'selected.json', {'epoch': epoch, 'proceed_to_test': proceed,
                            'checkpoint': None if path is None else str(path),
                            'sha256': None if path is None else hashlib.sha256(path.read_bytes()).hexdigest()})
        return summary


def test(output, workers=5):
    validate_workers(workers)
    output = Path(output)
    if (output / 'test_opened.json').exists():
        raise FileExistsError('test already opened')
    config = _registration(output)
    with run_lock(output, _digest(config)):
        if (output / 'test_opened.json').exists():
            raise FileExistsError('test already opened')
        selected = json.loads((output / 'selected.json').read_text(encoding='utf-8'))
        validation = json.loads((output / 'validation.json').read_text(encoding='utf-8'))
        if not selected.get('proceed_to_test') or not validation.get('proceed_to_test'):
            raise ValueError('VALIDATION_STOP')
        models = _candidate_paths(output, config)
        epoch = selected['epoch']
        if epoch != validation['selected_epoch'] or f'epoch_{epoch}' not in models:
            raise ValueError('selection does not match validation')
        path = models[f'epoch_{epoch}']
        if str(path) != selected['checkpoint'] or hashlib.sha256(path.read_bytes()).hexdigest() != selected['sha256']:
            raise ValueError('selected checkpoint mismatch')
        if (output / 'test_parts').exists():
            raise FileExistsError('test artifacts already exist')
        with (output / 'test_opened.json').open('x', encoding='utf-8') as handle:
            json.dump({'seed': config['test_seed'], 'boards': config['evaluation_boards'],
                       'selected_sha256': selected['sha256'], 'status': 'opened'}, handle)
        opponents = (*CNN_OPPONENTS, 'rulebot', 'S3', 'S4')
        arms = {'S4_11': path, 'S4': CHECKPOINTS['S4'], 'S3_11': CHECKPOINTS['S3_11']}
        results = _evaluation(output, 'test', arms, opponents, config['test_seed'], config, workers)
        comparison = _compare(results, 'S4_11', 'S4', opponents)
        summaries = _summaries(results, config['max_steps'])
        summary = {'seed': config['test_seed'], 'results': summaries,
                   'rates': {m: rates(summaries, m) for m in arms},
                   'S4_11_minus_S4': comparison, 'gate': gate(comparison), 'selected': selected}
        sd53._atomic_json(output / 'summary.json', summary)
        return summary


def main(argv=None):
    import argparse
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('init', 'pilot', 'collect', 'train', 'validate', 'test'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    actions = {'init': lambda: initialize(args.output, smoke=args.smoke),
               'pilot': lambda: pilot(args.output, args.workers),
               'collect': lambda: collect(args.output, args.workers, args.resume),
               'train': lambda: train(args.output, args.device),
               'validate': lambda: validate(args.output, args.workers),
               'test': lambda: test(args.output, args.workers)}
    result = actions[args.stage]()
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
