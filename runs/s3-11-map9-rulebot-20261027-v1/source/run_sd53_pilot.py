"""SD5.3: distil informative kill-search decisions into a pure 13-channel CNN.

Stages are intentionally separate and reproducible.  `collect` is the only
stage that creates an output directory; later stages reject a mismatched
registration.  The CNN used by validate/test never calls kill-search.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from benchmark_t3 import transform, transform_action
from progress import bar, parallel_map


ROOT = Path(__file__).resolve().parent
F = ROOT / 'runs/phaseC-sd-best-fallback/best_phaseC.pt'
F_SHA256 = '0315a56f09079333d936db5f55587093f1b6415ed6355d2ef6ae2ab63f8b1483'
SCHEDULE_E = dict(sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)
COLLECT_FIRST, COLLECT_LAST = 200000, 201499
VALIDATION_SEED, TEST_SEED, GAMES = 21000, 43000, 400
CANDIDATE_EPOCHS = (2, 4, 8)
SOURCE_FILES = ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py', 'evaluate.py',
                'kill_search.py', 'benchmark_t3.py', 'progress.py', 'pilot_two_bomb.py',
                'run_sd53_pilot.py')


def config_for(output: Path, smoke: bool) -> dict:
    """Frozen experiment settings; smoke only reduces workload, never protocol values."""
    boards = list(range(COLLECT_FIRST, COLLECT_LAST + 1))
    if smoke:
        boards = boards[:4]
    return {
        'protocol': 'SD5.3 dense search distillation', 'output': str(Path(output)), 'board_size': 9,
        # A 30-step smoke never reaches the combat states this experiment distils.
        'num_agents': 2, 'max_steps': 500, 'f_checkpoint': str(F),
        'f_sha256': F_SHA256, 'schedule_e': SCHEDULE_E, 'collect_first_seed': COLLECT_FIRST,
        'collect_last_seed': COLLECT_LAST, 'collect_board_seeds': boards, 'collect_games': len(boards) * 2,
        'schedule_e_probability': 0.75, 'anchor_probability': 0.05,
        'collection_shard_boards': 4 if smoke else 25, 'train_epochs': 2 if smoke else 8,
        'candidate_epochs': [2] if smoke else list(CANDIDATE_EPOCHS), 'learning_rate': 1e-4,
        'batch_size': 16 if smoke else 256, 'anchor_lambda': 1.0, 'heldout_fraction': 0.10,
        'validation_seed': VALIDATION_SEED, 'test_seed': TEST_SEED, 'evaluation_games': 4 if smoke else GAMES,
        'symmetry_min_samples': 8 if smoke else 200, 'symmetry_min_agreement': 0.90,
        'smoke': bool(smoke),
    }


def normalized_sha256(path: Path) -> str:
    """Hash source bytes after normalising CRLF and bare CR line endings to LF."""
    data = Path(path).read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    return hashlib.sha256(data).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def _runtime() -> dict:
    import torch
    return {'utc': datetime.now(timezone.utc).isoformat(), 'python': __import__('sys').version,
            'numpy': np.__version__, 'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def _require_f() -> None:
    if not F.is_file():
        raise FileNotFoundError(F)
    # Newline normalization applies to source provenance only, never binary checkpoints.
    if hashlib.sha256(F.read_bytes()).hexdigest() != F_SHA256:
        raise ValueError('F checkpoint changed; SD5.3 refuses to continue')


def _registration(output: Path, config: dict, create: bool) -> dict:
    output, marker = Path(output), Path(output) / 'registration.json'
    if create:
        output.mkdir(parents=True)  # FileExistsError is intentional: never overwrite a run.
        registration = {'config': config, 'F_sha256': F_SHA256,
                        'sources_normalized_lf_sha256': {name: normalized_sha256(ROOT / name) for name in SOURCE_FILES},
                        'runtime': _runtime(), 'symmetry': {'state': 'pending'}}
        _json(marker, registration)
        return registration
    if not marker.is_file():
        raise FileNotFoundError(f'missing registration: {marker}')
    registration = json.loads(marker.read_text(encoding='utf-8'))
    expected = {name: normalized_sha256(ROOT / name) for name in SOURCE_FILES}
    if registration.get('config') != config or registration.get('F_sha256') != F_SHA256:
        raise ValueError('output registration/configuration differs; resume refused')
    if registration.get('sources_normalized_lf_sha256') != expected:
        raise ValueError('source hash differs from registration; resume refused')
    return registration


def _stage(output: Path, config: dict, name: str) -> dict:
    _require_f()
    registration = _registration(output, config, create=False)
    _atomic_json(Path(output) / f'runtime_{name}.json', _runtime())
    return registration


def masked_soft_target(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Uniform mass over tied legal maxima, except an uninformative all-tie state."""
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if scores.shape != mask.shape:
        raise ValueError('scores and mask must have the same shape')
    target = np.zeros(scores.shape, dtype=np.float32)
    legal = mask & np.isfinite(scores)
    if not legal.any():
        return target
    best = scores[legal].max()
    # Collection records only differing score states.  Keep this guard so a
    # direct caller cannot turn an all-tie state into imitation noise.
    if np.all(scores[legal] == best):
        return target
    winners = legal & (scores == best)
    target[winners] = 1.0 / winners.sum()
    return target


def split_board_seeds(samples: np.ndarray, heldout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Partition sample seed IDs without ever separating corners of one board."""
    samples = np.asarray(samples)
    if samples.ndim != 1 or not len(samples):
        raise ValueError('samples must be a nonempty one-dimensional seed array')
    if not 0 < heldout_fraction < 1:
        raise ValueError('heldout_fraction must be between zero and one')
    heldout_seeds = _heldout_board_ids(samples, heldout_fraction, seed)
    heldout = samples[np.fromiter((int(item) in heldout_seeds for item in samples), dtype=bool, count=len(samples))]
    train = samples[np.fromiter((int(item) not in heldout_seeds for item in samples), dtype=bool, count=len(samples))]
    return train, heldout


def _heldout_board_ids(seeds: np.ndarray, heldout_fraction: float, seed: int) -> set[int]:
    unique = np.unique(np.asarray(seeds))
    if not len(unique):
        raise ValueError('at least one board seed is required')
    count = max(1, int(round(len(unique) * heldout_fraction)))
    return set(np.random.default_rng(seed).permutation(unique)[:count].tolist())


def augment_observation(obs: np.ndarray, rotation: int, mirror: bool) -> np.ndarray:
    obs = np.asarray(obs)
    if obs.ndim != 3 or obs.shape[1:] != (9, 9):
        raise ValueError('expected (channels, 9, 9) observation')
    out = np.empty_like(obs)
    for row in range(9):
        for col in range(9):
            transformed_row, transformed_col = transform((row, col), rotation, mirror)
            out[:, transformed_row, transformed_col] = obs[:, row, col]
    return out


def augment_action_vector(values: np.ndarray, rotation: int, mirror: bool) -> np.ndarray:
    values = np.asarray(values)
    out = np.empty_like(values)
    for action in range(len(values)):
        out[transform_action(action, rotation, mirror)] = values[action]
    return out


def _masked_logits(logits, mask):
    import torch
    valid = mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError('every sample needs one legal action')
    return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)


def _best_actions(scores: np.ndarray, mask: np.ndarray) -> set[int]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(scores)
    if not valid.any():
        return set()
    best = np.asarray(scores)[valid].max()
    return set(np.flatnonzero(valid & (np.asarray(scores) == best)).tolist())


def _schedule_e(seed: int) -> bool:
    return bool(np.random.default_rng(np.uint64(seed) ^ np.uint64(0x5D53)).random() < 0.75)


def _anchor(seed: int, aid: int, step: int) -> bool:
    return bool(np.random.default_rng(np.random.SeedSequence([seed, aid, step, 53])).random() < 0.05)


def _load_net(path: Path, device: str):
    import torch
    from train_actor_critic import ActorCriticNet, smart_load_weights
    net = ActorCriticNet(in_channels=13, board_size=9).to(device)
    smart_load_weights(net, str(path), device)
    return net.eval()


def _sample_f(net, obs: np.ndarray, mask: np.ndarray, device: str) -> tuple[int, np.ndarray]:
    import torch
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(obs).float().unsqueeze(0).to(device))
        masked = _masked_logits(logits, torch.from_numpy(mask).to(device).unsqueeze(0))
        action = int(torch.multinomial(torch.softmax(masked, dim=1), 1).item())
    return action, logits.squeeze(0).cpu().numpy().astype(np.float32)


def _transform_env(env, rotation: int, mirror: bool):
    transformed = deepcopy(env)
    transformed.grid = augment_observation(env.grid[None, ...], rotation, mirror)[0]
    for agent in transformed.agents:
        agent.row, agent.col = transform((agent.row, agent.col), rotation, mirror)
    for bomb in transformed.bombs:
        bomb.row, bomb.col = transform((bomb.row, bomb.col), rotation, mirror)
    transformed.flames = {transform(pos, rotation, mirror): value for pos, value in env.flames.items()}
    transformed._flame_owners = {transform(pos, rotation, mirror): value for pos, value in env._flame_owners.items()}
    transformed._flame_sources = {transform(pos, rotation, mirror): value for pos, value in env._flame_sources.items()}
    transformed.powerups = {transform(pos, rotation, mirror): value for pos, value in env.powerups.items()}
    return transformed


def _record_game(seed: int, aid: int, config: dict, net, device: str) -> tuple[dict, dict, list[dict], list[dict], list[bool]]:
    """One F+search game; returned records contain no live environment objects."""
    import torch
    from bomberman_env import BombermanFFAEnv
    from kill_search import search_action
    from rule_based_bot import RuleBasedAgent
    from train_actor_critic import get_action_mask
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_e = _schedule_e(seed)
    env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=config['max_steps'], seed=seed,
                          sudden_death=use_e, **SCHEDULE_E)
    obs_list, bots = env.reset(), {other: RuleBasedAgent(other) for other in range(2) if other != aid}
    decisions, anchors, audit = [], [], []
    suicide, sd_deaths = False, set()
    for step in range(config['max_steps']):
        if not env.agents[aid].alive:
            break
        obs, mask = np.asarray(obs_list[aid], dtype=np.float32), np.asarray(get_action_mask(env, aid), dtype=bool)
        sampled, logits = _sample_f(net, obs, mask, device)
        executed, search = search_action(env, aid, mask.tolist(), sampled)
        scores = np.full(len(mask), np.nan, dtype=np.float32)
        for action, score in search.get('scores', {}).items():
            scores[action] = score
        informative = not search.get('skipped', True) and len(set(search.get('scores', {}).values())) > 1
        base = {'obs': obs, 'mask': mask, 'seed': seed, 'corner': aid, 'step': step}
        if informative:
            decisions.append({**base, 'scores': scores, 'sampled_action': sampled, 'executed_action': executed})
            # One deterministic transform per state; the aggregate must reach the frozen sample floor.
            rotation, mirror = (seed + aid + step) % 4, bool((seed + step) & 1)
            transformed = _transform_env(env, rotation, mirror)
            transformed_mask = np.asarray(get_action_mask(transformed, aid), dtype=bool)
            _, transformed_search = search_action(transformed, aid, transformed_mask.tolist(),
                                                   transform_action(sampled, rotation, mirror))
            transformed_scores = np.full(len(mask), np.nan, dtype=np.float32)
            for action, score in transformed_search.get('scores', {}).items():
                transformed_scores[action] = score
            expected = {transform_action(action, rotation, mirror) for action in _best_actions(scores, mask)}
            audit.append(expected == _best_actions(transformed_scores, transformed_mask))
        elif _anchor(seed, aid, step):
            anchors.append({**base, 'logits': logits})
        actions = [0, 0]
        for other, bot in bots.items():
            if env.agents[other].alive:
                actions[other] = bot.select_action(env)
        actions[aid] = executed
        obs_list, _, _, info = env.step(actions)
        own_deaths = set(info.get('self_blast_deaths', [])) | {victim for killer, victim in info['kills'] if killer == victim}
        suicide |= aid in own_deaths
        sd_deaths.update(info.get('sudden_death_deaths', []))
        if info['episode_done']:
            break
    alive = [index for index, state in enumerate(info['alive']) if state]
    win = len(alive) == 1 and alive[0] == aid
    active = bool(win and env.agents[aid].kills > 0)
    return ({'seed': seed, 'aid': aid, 'win': win, 'active_win': active, 'draw': len(alive) != 1,
             'suicide': suicide, 'sudden_death_death': aid in sd_deaths, 'kills': env.agents[aid].kills,
             'steps': env.step_count, 'schedule_e': use_e}, {'decisions': decisions, 'anchors': anchors}, decisions, anchors, audit)


def _empty(shape, dtype):
    return np.empty((0, *shape), dtype=dtype)


def _write_shard(path: Path, metadata: dict, decisions: list[dict], anchors: list[dict], games: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'metadata': np.array(json.dumps(metadata)), 'games': np.array(json.dumps(games)),
        'decision_obs': np.stack([row['obs'] for row in decisions]).astype(np.float32) if decisions else _empty((13, 9, 9), np.float32),
        'decision_mask': np.stack([row['mask'] for row in decisions]).astype(bool) if decisions else _empty((6,), bool),
        'decision_scores': np.stack([row['scores'] for row in decisions]).astype(np.float32) if decisions else _empty((6,), np.float32),
        'decision_sampled_action': np.array([row['sampled_action'] for row in decisions], dtype=np.int8),
        'decision_executed_action': np.array([row['executed_action'] for row in decisions], dtype=np.int8),
        'decision_seed': np.array([row['seed'] for row in decisions], dtype=np.int64),
        'decision_corner': np.array([row['corner'] for row in decisions], dtype=np.int8),
        'decision_step': np.array([row['step'] for row in decisions], dtype=np.int16),
        'anchor_obs': np.stack([row['obs'] for row in anchors]).astype(np.float32) if anchors else _empty((13, 9, 9), np.float32),
        'anchor_mask': np.stack([row['mask'] for row in anchors]).astype(bool) if anchors else _empty((6,), bool),
        'anchor_logits': np.stack([row['logits'] for row in anchors]).astype(np.float32) if anchors else _empty((6,), np.float32),
        'anchor_seed': np.array([row['seed'] for row in anchors], dtype=np.int64),
        'anchor_corner': np.array([row['corner'] for row in anchors], dtype=np.int8),
        'anchor_step': np.array([row['step'] for row in anchors], dtype=np.int16),
    }
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def _collect_shard(start: int, end: int, output_text: str, config: dict, ticks) -> dict:
    import torch
    torch.set_num_threads(1)
    output = Path(output_text)
    net = _load_net(F, 'cpu')
    decisions, anchors, games, audit = [], [], [], []
    for seed in range(start, end + 1):
        for aid in (0, 1):
            game, _, found_decisions, found_anchors, found_audit = _record_game(seed, aid, config, net, 'cpu')
            games.append(game); decisions.extend(found_decisions); anchors.extend(found_anchors); audit.extend(found_audit)
            ticks.put(1)
    metadata = {'first_seed': start, 'last_seed': end, 'config_sha256': hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest(), 'source_sha256': {
            name: normalized_sha256(ROOT / name) for name in SOURCE_FILES}, 'F_sha256': F_SHA256,
        'audit_matches': audit}
    path = output / 'collection' / f'shard-{start}-{end}.npz'
    _write_shard(path, metadata, decisions, anchors, games)
    return {'path': str(path), 'games': len(games), 'decisions': len(decisions), 'anchors': len(anchors),
            'audit_samples': len(audit), 'audit_matches': int(sum(audit))}


def _valid_shard(path: Path, start: int, end: int, config: dict) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
        return (metadata['first_seed'] == start and metadata['last_seed'] == end and
                metadata['F_sha256'] == F_SHA256 and metadata['config_sha256'] == hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode()).hexdigest() and
                metadata['source_sha256'] == {name: normalized_sha256(ROOT / name) for name in SOURCE_FILES})
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _collect_summary(output: Path, config: dict) -> dict:
    games, audit = [], []
    for path in sorted((output / 'collection').glob('shard-*.npz')):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            games.extend(json.loads(str(data['games'])))
            audit.extend(metadata.get('audit_matches', []))
    if len(games) != config['collect_games']:
        raise ValueError('collection is incomplete')
    rates = {name: 100 * sum(bool(game[name]) for game in games) / len(games)
             for name in ('win', 'active_win', 'draw', 'suicide', 'sudden_death_death')}
    return {'games': len(games), 'diagnostic_rates': rates, 'audit_samples': len(audit),
            'audit_agreement': (sum(audit) / len(audit) if audit else 0.0), 'shards': len(list((output / 'collection').glob('shard-*.npz')))}


def collect(output: Path, smoke: bool = False, workers: int = 4, resume: bool = False) -> dict:
    _require_f()
    output, config = Path(output), config_for(output, smoke)
    if output.exists():
        if not resume:
            raise FileExistsError(f'{output} exists; use --resume only for an interrupted collect')
        registration = _stage(output, config, 'collect')
    else:
        registration = _registration(output, config, create=True)
    collection = output / 'collection'; collection.mkdir(exist_ok=True)
    seeds, width = config['collect_board_seeds'], config['collection_shard_boards']
    jobs = []
    for index in range(0, len(seeds), width):
        start, end = seeds[index], seeds[min(index + width - 1, len(seeds) - 1)]
        path = collection / f'shard-{start}-{end}.npz'
        if path.exists() and _valid_shard(path, start, end, config):
            continue
        if path.exists():
            raise ValueError(f'invalid completed shard: {path}; remove it explicitly before recollecting')
        jobs.append((start, end, str(output), config))
    if jobs:
        parallel_map(_collect_shard, jobs, total=sum((end - start + 1) * 2 for start, end, *_ in jobs),
                     desc='SD5.3 collect', workers=max(1, min(workers, len(jobs))))
    summary = _collect_summary(output, config)
    symmetry = {'samples': summary['audit_samples'], 'agreement': summary['audit_agreement'],
                'enabled': summary['audit_samples'] >= config['symmetry_min_samples'] and
                           summary['audit_agreement'] >= config['symmetry_min_agreement']}
    registration['symmetry'] = symmetry
    _atomic_json(output / 'registration.json', registration)
    _atomic_json(output / 'collection_summary.json', summary)
    _atomic_json(output / 'runtime_collect.json', _runtime())
    return summary


def _require_complete_collection(output: Path, config: dict, registration: dict) -> None:
    collection = output / 'collection'
    expected = []
    seeds, width = config['collect_board_seeds'], config['collection_shard_boards']
    for index in range(0, len(seeds), width):
        start, end = seeds[index], seeds[min(index + width - 1, len(seeds) - 1)]
        path = collection / f'shard-{start}-{end}.npz'
        if not path.is_file() or not _valid_shard(path, start, end, config):
            raise ValueError(f'collection incomplete or invalid at shard {start}-{end}')
        expected.append(path)
    actual = set(collection.glob('shard-*.npz'))
    if actual != set(expected):
        raise ValueError('collection has an unexpected shard set')
    symmetry = registration.get('symmetry', {})
    if symmetry.get('state') == 'pending' or 'enabled' not in symmetry:
        raise ValueError('collection symmetry audit is incomplete')


def _load_records(output: Path, config: dict) -> dict:
    keys = ('decision_obs', 'decision_mask', 'decision_scores', 'decision_seed', 'anchor_obs', 'anchor_mask',
            'anchor_logits', 'anchor_seed')
    chunks = {key: [] for key in keys}
    for path in sorted((output / 'collection').glob('shard-*.npz')):
        with np.load(path, allow_pickle=False) as data:
            for key in keys:
                chunks[key].append(data[key])
    if not chunks['decision_obs'] or not chunks['anchor_obs']:
        raise ValueError('collection records are missing')
    return {key: np.concatenate(parts) for key, parts in chunks.items()}


def _partition_indices(seeds: np.ndarray, held: set[int]) -> tuple[np.ndarray, np.ndarray]:
    return np.flatnonzero(~np.isin(seeds, list(held))), np.flatnonzero(np.isin(seeds, list(held)))


def _augment_batch(obs: np.ndarray, *vectors: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    outputs = [np.empty_like(obs), *[np.empty_like(vector) for vector in vectors]]
    for index in range(len(obs)):
        rotation, mirror = int(rng.integers(4)), bool(rng.integers(2))
        outputs[0][index] = augment_observation(obs[index], rotation, mirror)
        for target, vector in zip(outputs[1:], vectors):
            target[index] = augment_action_vector(vector[index], rotation, mirror)
    return tuple(outputs)


def _heldout_metrics(net, records: dict, decision_indices: np.ndarray, anchor_indices: np.ndarray, device: str) -> dict:
    import torch
    import torch.nn.functional as functional
    if not len(decision_indices) or not len(anchor_indices):
        raise ValueError('held-out split needs decision and anchor samples')
    with torch.no_grad():
        decision_obs = torch.from_numpy(records['decision_obs'][decision_indices]).float().to(device)
        decision_mask = torch.from_numpy(records['decision_mask'][decision_indices]).to(device)
        decision_logits, _ = net(decision_obs)
        choice = _masked_logits(decision_logits, decision_mask).argmax(1).cpu().numpy()
        targets = np.stack([masked_soft_target(scores, mask) for scores, mask in zip(
            records['decision_scores'][decision_indices], records['decision_mask'][decision_indices])])
        useful = targets.sum(1) > 0
        correct = targets[np.arange(len(targets)), choice] > 0
        anchor_obs = torch.from_numpy(records['anchor_obs'][anchor_indices]).float().to(device)
        anchor_mask = torch.from_numpy(records['anchor_mask'][anchor_indices]).to(device)
        model_logits, _ = net(anchor_obs)
        model_log = functional.log_softmax(_masked_logits(model_logits, anchor_mask), dim=1)
        teacher = torch.from_numpy(records['anchor_logits'][anchor_indices]).float().to(device)
        teacher_prob = functional.softmax(_masked_logits(teacher, anchor_mask), dim=1)
        kl = functional.kl_div(model_log, teacher_prob, reduction='batchmean').item()
        teacher_choice = _masked_logits(teacher, anchor_mask).argmax(1)
        changed = (model_log.argmax(1) != teacher_choice).float().mean().item()
    kill = np.nan_to_num(records['decision_scores'][decision_indices], nan=-np.inf).max(1) > 0
    return {'decision_argmax_best_rate': float(correct[useful].mean() * 100) if useful.any() else None,
            'kill_decision_argmax_best_rate': float(correct[useful & kill].mean() * 100) if (useful & kill).any() else None,
            'anchor_kl': float(kl), 'anchor_argmax_changed_rate': float(changed * 100),
            'heldout_decisions': int(len(decision_indices)), 'heldout_anchors': int(len(anchor_indices))}


def train(output: Path, smoke: bool = False, device: str = 'cpu') -> dict:
    import torch
    import torch.nn.functional as functional
    _require_f()
    output, config = Path(output), config_for(output, smoke)
    registration = _stage(output, config, 'train')
    _require_complete_collection(output, config, registration)
    records = _load_records(output, config)
    decision_target = np.stack([masked_soft_target(scores, mask) for scores, mask in zip(
        records['decision_scores'], records['decision_mask'])])
    valid_decisions = np.flatnonzero(decision_target.sum(1) > 0)
    if not len(valid_decisions):
        raise ValueError('collection contains no positive informative decision target')
    heldout_boards = _heldout_board_ids(np.concatenate((records['decision_seed'], records['anchor_seed'])),
                                        config['heldout_fraction'], 530)
    decision_train_all, decision_holdout = _partition_indices(records['decision_seed'], heldout_boards)
    anchor_train, anchor_holdout = _partition_indices(records['anchor_seed'], heldout_boards)
    decision_train = np.intersect1d(decision_train_all, valid_decisions, assume_unique=False)
    if not len(decision_train) or not len(anchor_train) or not len(decision_holdout) or not len(anchor_holdout):
        raise ValueError('insufficient records after board-seed split')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    net = _load_net(F, device).train()
    optimizer = torch.optim.Adam([parameter for name, parameter in net.named_parameters() if not name.startswith('critic_head.')],
                                 lr=config['learning_rate'])
    rng = np.random.default_rng(530)
    destination = output / 'train'
    destination.mkdir()  # no silent reuse
    metrics = []
    half = config['batch_size'] // 2
    batches = max(int(np.ceil(len(decision_train) / half)), int(np.ceil(len(anchor_train) / half)))
    with bar(config['train_epochs'] * batches, 'SD5.3 train') as shown:
      for epoch in range(1, config['train_epochs'] + 1):
        order_d, order_a = rng.permutation(decision_train), rng.permutation(anchor_train)
        losses = []
        for batch in range(batches):
            di = order_d[(np.arange(half) + batch * half) % len(order_d)]
            ai = order_a[(np.arange(half) + batch * half) % len(order_a)]
            d_obs, d_mask, d_target = records['decision_obs'][di], records['decision_mask'][di], decision_target[di]
            a_obs, a_mask, a_logits = records['anchor_obs'][ai], records['anchor_mask'][ai], records['anchor_logits'][ai]
            if registration['symmetry'].get('enabled'):
                d_obs, d_mask, d_target = _augment_batch(d_obs, d_mask, d_target, rng=rng)
                a_obs, a_mask, a_logits = _augment_batch(a_obs, a_mask, a_logits, rng=rng)
            d_logits, _ = net(torch.from_numpy(d_obs).float().to(device))
            d_log = functional.log_softmax(_masked_logits(d_logits, torch.from_numpy(d_mask).to(device)), dim=1)
            ce = -(torch.from_numpy(d_target).float().to(device) * d_log).sum(1).mean()
            a_model, _ = net(torch.from_numpy(a_obs).float().to(device))
            a_mask_t = torch.from_numpy(a_mask).to(device)
            a_log = functional.log_softmax(_masked_logits(a_model, a_mask_t), dim=1)
            a_teacher = torch.from_numpy(a_logits).float().to(device)
            kl = functional.kl_div(a_log, functional.softmax(_masked_logits(a_teacher, a_mask_t), dim=1), reduction='batchmean')
            loss = ce + config['anchor_lambda'] * kl
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.item())); shown.update(1)
        net.eval()
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
               **_heldout_metrics(net, records, decision_holdout, anchor_holdout, device)}
        metrics.append(row)
        if epoch in config['candidate_epochs']:
            torch.save(net.state_dict(), destination / f'epoch_{epoch}.pt')
        net.train()
    _atomic_json(destination / 'metrics.json', {'metrics': metrics, 'split': {
        'decision_train': int(len(decision_train)), 'decision_heldout': int(len(decision_holdout)),
        'anchor_train': int(len(anchor_train)), 'anchor_heldout': int(len(anchor_holdout))}})
    _atomic_json(destination / 'provenance.json', {'F_sha256': F_SHA256, 'registration_sources': registration['sources_normalized_lf_sha256']})
    _atomic_json(output / 'runtime_train.json', _runtime())
    return {'metrics': metrics, 'train_dir': str(destination)}


def _evaluate_checkpoint(path: str, rule: str, seed: int, games: int, max_steps: int, ticks):
    import torch
    import evaluate
    torch.set_num_threads(1)
    done = [0]
    def advance(value, total):
        ticks.put(value - done[0]); done[0] = value
    return evaluate.evaluate_seeded(evaluate.make_ac_fn(path, 9), 9, 2, games, seed, max_steps,
                                    on_progress=advance, sudden_death=rule == 'E', **SCHEDULE_E)


def _evaluate_search(rule: str, seed: int, games: int, max_steps: int, ticks):
    import torch
    import evaluate
    from kill_search import search_action
    from train_actor_critic import get_action_mask
    torch.set_num_threads(1)
    policy = evaluate.make_ac_fn(str(F), 9)
    done = [0]
    def advance(value, total):
        ticks.put(value - done[0]); done[0] = value
    agent = lambda env, obs, aid: search_action(env, aid, get_action_mask(env, aid), policy(env, obs, aid))[0]
    return evaluate.evaluate_seeded(agent, 9, 2, games, seed, max_steps, on_progress=advance,
                                    sudden_death=rule == 'E', **SCHEDULE_E)


def _paired(left: dict, right: dict) -> dict:
    from pilot_two_bomb import paired_active_bootstrap
    return {metric: paired_active_bootstrap(*({'games': [{**game, 'active_win': game[metric]} for game in result['games']]}
                                               for result in (left, right)))
            for metric in ('active_win', 'win', 'draw', 'suicide')}


def _safe_candidates(results: dict) -> list[tuple[int, dict]]:
    baseline = results['F']['suicide_rate']
    return [(int(name.removeprefix('epoch_')), result) for name, result in results.items() if name != 'F'
            and result['suicide_rate'] <= baseline + 1.0]


def validate(output: Path, smoke: bool = False, workers: int = 3) -> dict:
    _require_f()
    output, config = Path(output), config_for(output, smoke)
    _stage(output, config, 'validate')
    if (output / 'test_opened.json').exists():
        raise ValueError('validation is frozen after opening the test seed')
    if (output / 'validation.json').exists() or (output / 'selected.json').exists():
        raise FileExistsError('validation already exists; selection cannot be overwritten')
    paths = {'F': F, **{f'epoch_{epoch}': output / 'train' / f'epoch_{epoch}.pt' for epoch in config['candidate_epochs']}}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing validation models: {missing}')
    jobs = [(str(path), 'E', VALIDATION_SEED, config['evaluation_games'], config['max_steps']) for path in paths.values()]
    values = parallel_map(_evaluate_checkpoint, jobs, total=len(jobs) * config['evaluation_games'],
                          desc=f'SD5.3 validate seed {VALIDATION_SEED}', workers=max(1, min(workers, len(jobs))))
    results = dict(zip(paths, values))
    safe = _safe_candidates(results)
    if not safe:
        raise ValueError('no candidate satisfies validation suicide constraint')
    epoch, selected_result = max(safe, key=lambda pair: (pair[1]['active_win_rate'], -pair[0]))
    paired = {name: _paired(result, results['F']) for name, result in results.items() if name != 'F'}
    summary = {'seed': VALIDATION_SEED, 'games': config['evaluation_games'], 'rule': 'E', 'results': results,
               'paired_vs_f': paired, 'selected_epoch': epoch, 'selected_checkpoint': str(paths[f'epoch_{epoch}']),
               'selection': {'criterion': 'highest ActiveWR with suicide <= F + 1pp',
                             'active_win_rate': selected_result['active_win_rate']}}
    _atomic_json(output / 'validation.json', summary)
    _atomic_json(output / 'selected.json', {'epoch': epoch, 'checkpoint': str(paths[f'epoch_{epoch}']),
                                             'selection': summary['selection']})
    return summary


def gate(paired: dict) -> dict:
    vs_f = paired['search_vs_f']
    checks = {'active_vs_f_5pp': vs_f['active_win']['difference_pp'] >= 5.0,
              'active_vs_f_ci_lower_positive': vs_f['active_win']['ci95_pp'][0] > 0,
              'suicide_within_1pp': vs_f['suicide']['difference_pp'] <= 1.0}
    return {**checks, 'pass': all(checks.values())}


def test(output: Path, smoke: bool = False, workers: int = 3) -> dict:
    output = Path(output)
    opened = output / 'test_opened.json'
    if opened.exists():
        raise FileExistsError(f'test seed already opened: {opened}')
    _require_f()
    config = config_for(output, smoke)
    _stage(output, config, 'test')
    selected = json.loads((output / 'selected.json').read_text(encoding='utf-8'))
    candidate = Path(selected['checkpoint'])
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    paths = {'F': F, 'sd53_best': candidate}
    with open(opened, 'x', encoding='utf-8') as handle:  # exclusive before any evaluation
        json.dump({'seed': TEST_SEED, 'games': config['evaluation_games'],
                   'models': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}}, handle, indent=2)
    jobs, names = [], []
    for name, path in paths.items():
        for rule in ('E', 'off'):
            names.append(f'{name}_{rule}')
            jobs.append((_evaluate_checkpoint, (str(path), rule, TEST_SEED, config['evaluation_games'], config['max_steps'])))
    for rule in ('E', 'off'):
        names.append(f'F_search_{rule}')
        jobs.append((_evaluate_search, (rule, TEST_SEED, config['evaluation_games'], config['max_steps'])))
    # parallel_map needs one picklable function; evaluate checkpoint jobs share it, search is sequentially reported.
    checkpoint_jobs = [job for function, job in jobs if function is _evaluate_checkpoint]
    checkpoint_values = parallel_map(_evaluate_checkpoint, checkpoint_jobs, total=len(checkpoint_jobs) * config['evaluation_games'],
                                     desc=f'SD5.3 test seed {TEST_SEED}', workers=max(1, min(workers, len(checkpoint_jobs))))
    results = {name: value for name, value in zip([name for name, (function, _) in zip(names, jobs) if function is _evaluate_checkpoint], checkpoint_values)}
    # Search is a report-only ceiling and intentionally never enters the selection/gate calculations.
    search_jobs = [(name, job) for name, (function, job) in zip(names, jobs) if function is _evaluate_search]
    if search_jobs:
        with bar(len(search_jobs) * config['evaluation_games'], 'SD5.3 test F+search') as shown:
            class _Ticks:
                def put(self, value):
                    shown.update(value)
            for name, job in search_jobs:
                results[name] = _evaluate_search(*job, _Ticks())
    for name, value in results.items():
        _atomic_json(output / f'evaluation_{name}_{TEST_SEED}.json', value)
    paired = {rule: {'search_vs_f': _paired(results[f'sd53_best_{rule}'], results[f'F_{rule}'])}
              for rule in ('E', 'off')}
    summary = {'seed': TEST_SEED, 'games': config['evaluation_games'], 'results': results,
               'paired': paired, 'gate': gate(paired['E'])}
    _atomic_json(output / 'summary.json', summary)
    shutil.make_archive(str(output) + '-send', 'zip', output, '.')
    return summary


def main(argv=None):
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('collect', 'train', 'validate', 'test'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true', help='collect only')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if args.resume and args.stage != 'collect':
        parser.error('--resume is only valid for collect')
    result = {'collect': lambda: collect(args.output, args.smoke, args.workers, args.resume),
              'train': lambda: train(args.output, args.smoke, args.device),
              'validate': lambda: validate(args.output, args.smoke, args.workers),
              'test': lambda: test(args.output, args.smoke, args.workers)}[args.stage]()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
