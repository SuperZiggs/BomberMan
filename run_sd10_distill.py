"""SD10 (direction A): distil S3 + three-move kill-search that models each opponent correctly into a pure CNN (S4).

Spec: docs/superpowers/specs/2026-10-24-sd10-cnn-aware-distillation.md (rules fixed before running); commands in
SD10_GUIDE.md. Same pipeline as SD9 (`run_sd9_distill.py`, left unchanged because its finished run checks its own
source hash), with these changes:

  collect   2000 games of S3+search3 on boards 214000-214999, opponent by board: RuleBot, S2, G, S3 in turn
            (CNN opponents sample). Against a CNN the search predicts it with that CNN's own argmax policy
            (`probe_search_cnn_model.cnn_model`, SDM's oracle); against RuleBot it predicts RuleBot as before.
            Decision states / anchors as in SD9; anchors keep S3's logits.
  train     from S3, soft-target CE on decision states + KL to S3 on anchors (lambda 1), 16 epochs, candidates
            4/8/12/16. S4 plays as a plain CNN: the opponent model is only used to make labels.
  validate  S3 and the candidates vs S2, G, S3 and RuleBot on seed 21000 (400 games each). Eligible: RuleBot
            ActiveWR >= S3 - 3pp, suicide (macro CNN and RuleBot) <= S3 + 1pp. Select the highest macro-CNN
            ActiveWR (ties: earlier epoch). The test opens only if it beats S3 on validation (else VALIDATION_STOP).
  test      seed 58000, once: S4 and S3 vs S2, G, S3, RuleBot and F (F unseen, report only).
            Gate: macro CNN ActiveWR S4 - S3 >= +5pp with CI95 lower > 0, macro CNN suicide <= +1pp,
            RuleBot ActiveWR S4 - S3 >= -3pp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from progress import bar, parallel_map
import probe_s3_cnn as sd91
import run_pde_check as pde
import run_sd53_pilot as sd53
from run_sd9_distill import teacher_scores

ROOT = Path(__file__).resolve().parent
S3, S3_SHA256 = pde.CKPT['S3'], sd91.SHA256['S3']
SHA256 = {**sd91.SHA256, 'F': pde.sd5.F_SHA256}
SCHEDULE_E = sd53.SCHEDULE_E
COLLECT_FIRST, COLLECT_LAST = 214000, 214999
COLLECT_OPPONENTS = ('rulebot', 'S2', 'G', 'S3')
EVAL_OPPONENTS = ('S2', 'G', 'S3', 'rulebot')
TEST_OPPONENTS = EVAL_OPPONENTS + ('F',)
VALIDATION_SEED, TEST_SEED, BOARDS, MAX_STEPS = 21000, 58000, 200, 500
METRICS = ('active_win', 'win', 'draw', 'passive_win', 'suicide')
SOURCE_FILES = ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py', 'evaluate.py', 'kill_search.py',
                'progress.py', 'pilot_two_bomb.py', 'run_sd53_pilot.py', 'run_sd9_distill.py', 'probe_s3_cnn.py',
                'probe_search_cnn_model.py', 'run_pde_check.py', 'run_sd10_distill.py')


def config_for(output, smoke):
    boards = list(range(COLLECT_FIRST, COLLECT_LAST + 1))[:4 if smoke else None]
    return {'protocol': 'SD10 opponent-aware search3 distillation from S3', 'output': str(Path(output)),
            'max_steps': 60 if smoke else MAX_STEPS, 'sha256': SHA256, 'schedule_e': SCHEDULE_E,
            'collect_board_seeds': boards, 'collect_games': len(boards) * 2, 'collect_opponents': COLLECT_OPPONENTS,
            'schedule_e_probability': 0.75, 'anchor_probability': 0.05, 'collection_shard_boards': 2 if smoke else 20,
            'train_epochs': 2 if smoke else 16, 'candidate_epochs': [2] if smoke else [4, 8, 12, 16],
            'learning_rate': 1e-4, 'batch_size': 16 if smoke else 256, 'anchor_lambda': 1.0,
            'heldout_fraction': 0.10, 'validation_seed': VALIDATION_SEED, 'test_seed': TEST_SEED,
            'evaluation_boards': 2 if smoke else BOARDS, 'evaluation_shard_boards': 1 if smoke else 20,
            'smoke': bool(smoke)}


def opponent_for(seed):
    return COLLECT_OPPONENTS[(seed - COLLECT_FIRST) % len(COLLECT_OPPONENTS)]


def _sources():
    return {name: sd53.normalized_sha256(ROOT / name) for name in SOURCE_FILES}


def _check_checkpoints(keys=('S2', 'G', 'S3')):
    for key in keys:
        path = pde.CKPT[key]
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SHA256[key]:
            raise ValueError(f'checkpoint changed: {path}')


def _registration(output, config, create=False):
    marker = Path(output) / 'registration.json'
    if create:
        Path(output).mkdir(parents=True)  # FileExistsError: never overwrite a run
        registration = {'config': config, 'sources_normalized_lf_sha256': _sources(), 'runtime': sd53._runtime()}
        sd53._atomic_json(marker, registration)
        return registration
    registration = json.loads(marker.read_text(encoding='utf-8'))
    if registration['config'] != json.loads(json.dumps(config)):
        raise ValueError('configuration differs from registration; refused')
    if registration['sources_normalized_lf_sha256'] != _sources():
        raise ValueError('source hash differs from registration; refused')
    return registration


def _opponent_models():
    """name -> (sampled policy fn(env, obs_list, aid), argmax model fn(env, aid) for the search)."""
    import evaluate
    from probe_search_cnn_model import cnn_model
    return {name: (evaluate.make_ac_fn(str(pde.CKPT[name]), 9), cnn_model(pde.CKPT[name], 9))
            for name in COLLECT_OPPONENTS if name != 'rulebot'}


def _record_game(seed, aid, config, net, models):
    import torch
    import kill_search
    from bomberman_env import BombermanFFAEnv
    from rule_based_bot import RuleBasedAgent
    from train_actor_critic import get_action_mask
    np.random.seed(seed)
    torch.manual_seed(seed)
    opponent, other = opponent_for(seed), 1 - aid
    use_e = sd53._schedule_e(seed)
    env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=config['max_steps'], seed=seed,
                          sudden_death=use_e, **SCHEDULE_E)
    obs_list = env.reset()
    bot = RuleBasedAgent(other) if opponent == 'rulebot' else None
    kill_search.OPPONENT_MODEL = None if bot else models[opponent][1]
    decisions, anchors, info = [], [], {'alive': [True, True]}
    try:
        for step in range(config['max_steps']):
            if not env.agents[aid].alive:
                break
            obs, mask = np.asarray(obs_list[aid], dtype=np.float32), np.asarray(get_action_mask(env, aid), dtype=bool)
            sampled, logits = sd53._sample_f(net, obs, mask, 'cpu')
            executed, search = kill_search.search_action3(env, aid, mask.tolist(), sampled)
            scores = teacher_scores(search, len(mask))
            base = {'obs': obs, 'mask': mask, 'seed': seed, 'corner': aid, 'step': step}
            if not search.get('skipped', True) and len(set(scores[~np.isnan(scores)].tolist())) > 1:
                decisions.append({**base, 'scores': scores, 'sampled_action': sampled, 'executed_action': executed})
            elif sd53._anchor(seed, aid, step):
                anchors.append({**base, 'logits': logits})
            actions = [0, 0]
            if env.agents[other].alive:
                actions[other] = bot.select_action(env) if bot else models[opponent][0](env, obs_list, other)
            actions[aid] = executed
            obs_list, _, _, info = env.step(actions)
            if info['episode_done']:
                break
    finally:
        kill_search.OPPONENT_MODEL = None
    alive = [i for i, state in enumerate(info['alive']) if state]
    win = len(alive) == 1 and alive[0] == aid
    game = {'seed': seed, 'aid': aid, 'opponent': opponent, 'win': win,
            'active_win': bool(win and env.agents[aid].kills > 0), 'draw': len(alive) != 1,
            'steps': env.step_count, 'schedule_e': use_e, 'decisions': len(decisions)}
    return game, decisions, anchors


def _shard_meta(start, end, config):
    return {'first_seed': start, 'last_seed': end, 'sha256': SHA256, 'sources': _sources(),
            'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}


def _collect_shard(start, end, output, config, ticks):
    import torch
    torch.set_num_threads(1)
    net, models = sd53._load_net(S3, 'cpu'), _opponent_models()
    games, decisions, anchors = [], [], []
    for seed in range(start, end + 1):
        for aid in (0, 1):
            game, found, anchored = _record_game(seed, aid, config, net, models)
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
            return json.loads(str(data['metadata'])) == json.loads(json.dumps(_shard_meta(start, end, config)))
    except (OSError, KeyError, ValueError):
        return False


def collect(output, smoke=False, workers=5, resume=False):
    _check_checkpoints()
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
                     desc='SD10 collect (S3+search3, opponent-aware)', workers=max(1, min(workers, len(jobs))))
    games, decisions, anchors = [], 0, 0
    for path in sorted((output / 'collection').glob('shard-*.npz')):
        with np.load(path, allow_pickle=False) as data:
            games += json.loads(str(data['games'])); decisions += len(data['decision_seed']); anchors += len(data['anchor_seed'])
    if len(games) != config['collect_games']:
        raise ValueError('collection is incomplete')
    by_opponent = {}
    for o in COLLECT_OPPONENTS:
        rows = [g for g in games if g['opponent'] == o]
        by_opponent[o] = {'games': len(rows), 'decisions': sum(g['decisions'] for g in rows),
                          **{f'{k}_rate': 100 * sum(bool(g[k]) for g in rows) / max(1, len(rows))
                             for k in ('win', 'active_win', 'draw')}}
    summary = {'games': len(games), 'decisions': decisions, 'anchors': anchors, 'by_opponent': by_opponent}
    sd53._atomic_json(output / 'collection_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
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
    _check_checkpoints(('S3',))
    output, config = Path(output), config_for(output, smoke)
    _registration(output, config)
    records = _load_records(output, config)
    target = np.stack([sd53.masked_soft_target(s, m) for s, m in zip(records['decision_scores'], records['decision_mask'])])
    held = sd53._heldout_board_ids(np.concatenate((records['decision_seed'], records['anchor_seed'])),
                                   config['heldout_fraction'], 1010)
    d_train, d_held = sd53._partition_indices(records['decision_seed'], held)
    a_train, a_held = sd53._partition_indices(records['anchor_seed'], held)
    d_train = np.intersect1d(d_train, np.flatnonzero(target.sum(1) > 0))
    if not (len(d_train) and len(a_train) and len(d_held) and len(a_held)):
        raise ValueError('insufficient records after the board-seed split')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    net = sd53._load_net(S3, device).train()
    optimizer = torch.optim.Adam([p for n, p in net.named_parameters() if not n.startswith('critic_head.')],
                                 lr=config['learning_rate'])
    rng, half = np.random.default_rng(1010), config['batch_size'] // 2
    destination = output / 'train'
    destination.mkdir()  # no silent reuse
    batches = max(-(-len(d_train) // half), -(-len(a_train) // half))
    as_tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)
    metrics = [{'epoch': 0, **sd53._heldout_metrics(net.eval(), records, d_held, a_held, device)}]
    net.train()
    with bar(config['train_epochs'] * batches, 'SD10 train') as shown:
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


def play(name, path, opponent, seed, boards, max_steps, ticks):
    """Pure CNN `path` (sampled) vs `opponent` on boards seed..seed+boards-1, both corners, schedule E."""
    import torch
    torch.set_num_threads(1)
    import evaluate
    done = [0]

    def advance(value, total):
        ticks.put(value - done[0]); done[0] = value
    result = evaluate.evaluate_seeded(evaluate.make_ac_fn(str(path), 9), 9, 2, boards * 2, seed, max_steps,
                                      on_progress=advance, sudden_death=True, opponent_fn=pde._seat(opponent, 9),
                                      **SCHEDULE_E)
    return f'{name}_vs_{opponent}', result['games']


def run_jobs(models, opponents, seed, config, workers, desc):
    boards, width = config['evaluation_boards'], config['evaluation_shard_boards']
    jobs = [(name, str(path), o, seed + first, min(width, boards - first), config['max_steps'])
            for name, path in models.items() for o in opponents for first in range(0, boards, width)]
    parts = parallel_map(play, jobs, total=sum(j[4] * 2 for j in jobs), desc=desc, workers=workers)
    results = {}
    for key, games in parts:
        results.setdefault(key, []).extend(games)
    return {k: sorted(v, key=lambda g: (g['seed'], g['aid'])) for k, v in results.items()}


def compare(results, a, b, opponents):
    value = lambda rows, m: [{**g, 'value': float(g[m])} for g in rows]
    out = {'macro_cnn': {m: pde.paired_bootstrap(sd91._macro(results, a, m), sd91._macro(results, b, m)) for m in METRICS}}
    out.update({o: {m: pde.paired_bootstrap(value(results[f'{a}_vs_{o}'], m), value(results[f'{b}_vs_{o}'], m))
                    for m in METRICS} for o in opponents})
    return out


def rates(summaries, name):
    """Macro-CNN rates (mean over S2/G/S3, equal game counts) and RuleBot rates of one model."""
    macro = {k: float(np.mean([summaries[f'{name}_vs_{o}'][k] for o in sd91.CNN_OPPONENTS]))
             for k in ('active_win_rate', 'suicide_rate', 'draw_rate', 'win_rate')}
    return {'macro_cnn': macro, 'rulebot': summaries[f'{name}_vs_rulebot']}


def select(summaries, epochs):
    """Rule fixed in the spec. Returns (epoch or None, proceed_to_test, per-candidate checks)."""
    base, checks = rates(summaries, 'S3'), {}
    for e in epochs:
        r = rates(summaries, f'epoch_{e}')
        checks[e] = {'rulebot_active_within_3pp': r['rulebot']['active_win_rate'] >= base['rulebot']['active_win_rate'] - 3,
                     'macro_suicide_within_1pp': r['macro_cnn']['suicide_rate'] <= base['macro_cnn']['suicide_rate'] + 1,
                     'rulebot_suicide_within_1pp': r['rulebot']['suicide_rate'] <= base['rulebot']['suicide_rate'] + 1,
                     'macro_active': r['macro_cnn']['active_win_rate']}
    eligible = [e for e in epochs if all(v for k, v in checks[e].items() if k != 'macro_active')]
    if not eligible:
        return None, False, checks
    epoch = max(eligible, key=lambda e: (checks[e]['macro_active'], -e))
    return epoch, checks[epoch]['macro_active'] > base['macro_cnn']['active_win_rate'], checks


def validate(output, smoke=False, workers=5):
    from run_sd78_search_test import summarize
    _check_checkpoints()
    output, config = Path(output), config_for(output, smoke)
    _registration(output, config)
    if (output / 'selected.json').exists() or (output / 'test_opened.json').exists():
        raise FileExistsError('validation already done; selection cannot be overwritten')
    models = {'S3': S3, **{f'epoch_{e}': output / 'train' / f'epoch_{e}.pt' for e in config['candidate_epochs']}}
    results = run_jobs(models, EVAL_OPPONENTS, VALIDATION_SEED, config, workers, f'SD10 validate {VALIDATION_SEED}')
    summaries = {k: summarize(v) for k, v in results.items()}
    epoch, proceed, checks = select(summaries, config['candidate_epochs'])
    summary = {'seed': VALIDATION_SEED, 'results': summaries, 'rates': {m: rates(summaries, m) for m in models},
               'paired_vs_S3': {m: compare(results, m, 'S3', EVAL_OPPONENTS) for m in models if m != 'S3'},
               'checks': checks, 'selected_epoch': epoch, 'proceed_to_test': proceed}
    sd53._atomic_json(output / 'validation.json', summary)
    sd53._atomic_json(output / 'selected.json', {'epoch': epoch, 'proceed_to_test': proceed, 'checkpoint':
                                                 None if epoch is None else str(output / 'train' / f'epoch_{epoch}.pt')})
    print(json.dumps({k: summary[k] for k in ('rates', 'checks', 'selected_epoch', 'proceed_to_test')}, indent=2), flush=True)
    return summary


def gate(cmp):
    macro = cmp['macro_cnn']
    checks = {'macro_active_5pp': macro['active_win']['difference_pp'] >= 5.0,
              'macro_active_ci_lower_positive': macro['active_win']['ci95_pp'][0] > 0,
              'macro_suicide_within_1pp': macro['suicide']['difference_pp'] <= 1.0,
              'rulebot_active_within_3pp': cmp['rulebot']['active_win']['difference_pp'] >= -3.0}
    return {**checks, 'pass': all(checks.values())}


def test(output, smoke=False, workers=5):
    from run_sd78_search_test import summarize
    output, config = Path(output), config_for(output, smoke)
    _registration(output, config)
    _check_checkpoints(('S2', 'G', 'S3', 'F'))
    selected = json.loads((output / 'selected.json').read_text(encoding='utf-8'))
    if not selected['proceed_to_test']:
        raise ValueError('VALIDATION_STOP: no candidate beat S3 on validation; the test seed stays unopened')
    path = Path(selected['checkpoint'])
    seed = VALIDATION_SEED if smoke else TEST_SEED
    with open(output / 'test_opened.json', 'x', encoding='utf-8') as handle:  # the test seed is used once
        json.dump({'seed': seed, 'boards': config['evaluation_boards'],
                   'models': {'S4': hashlib.sha256(path.read_bytes()).hexdigest(), **SHA256}}, handle, indent=2)
    results = run_jobs({'S4': path, 'S3': S3}, TEST_OPPONENTS, seed, config, workers, f'SD10 test seed {seed}')
    for key, rows in results.items():
        (output / f'evaluation_{key}_{seed}.json').write_text(json.dumps({'games': rows}), encoding='utf-8')
    cmp = compare(results, 'S4', 'S3', TEST_OPPONENTS)
    summaries = {k: summarize(v) for k, v in results.items()}
    summary = {'seed': seed, 'boards': config['evaluation_boards'], 'selected': str(path), 'results': summaries,
               'rates': {m: rates(summaries, m) for m in ('S4', 'S3')}, 'S4_minus_S3': cmp, 'gate': gate(cmp)}
    sd53._atomic_json(output / 'summary.json', summary)
    if not smoke:
        shutil.make_archive(str(output) + '-send', 'zip', output, '.')
    print(json.dumps({k: summary[k] for k in ('rates', 'gate')}, indent=2), flush=True)
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
    {'collect': lambda: collect(a.output, a.smoke, a.workers, a.resume),
     'train': lambda: train(a.output, a.smoke, a.device),
     'validate': lambda: validate(a.output, a.smoke, a.workers),
     'test': lambda: test(a.output, a.smoke, a.workers)}[a.stage]()


if __name__ == '__main__':
    main()
