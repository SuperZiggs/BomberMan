"""Post-hoc analysis of committed D-S2 data; never rolls out or updates a model."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUN = REPO / 'runs/phaseD-s2-20261022'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RUN / 'sources'))

from phase_d_s2_protocol import CNN, canonical_digest, read_json, sha, verify_run
from phase_d_s2_evaluation import (
    candidates_for_run, load_registered_model, paired_macro_comparison,
    summarize, validation_tasks,
)
from phase_d_s2_ledger import _read_ledger, _rows, _selection, _validation_report
from phase_d_s2_training import build_train_schedule
from training_state import load_training_state
from bomberman_env import ACTION_BOMB


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def table(name, rows):
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rate(rows, key):
    return float(np.mean([r[key] for r in rows]) * 100)


def training_summary(rows):
    result = dict(episodes=len(rows), active_wr=rate(rows, 'active_win'), wr=rate(rows, 'win'),
                  draw=rate(rows, 'draw'), suicide=rate(rows, 'suicide'))
    for k in ('reward', 'learner_steps', 'bombs', 'walls', 'kills', 'value_loss', 'policy_loss',
              'combat_loss', 'navigation_loss', 'escape_loss', 'loss'):
        result[k] = float(np.mean([r[k] for r in rows]))
    result['total_steps'] = sum(r['learner_steps'] for r in rows)
    result['bombs_per_100_steps'] = 100 * sum(r['bombs'] for r in rows) / result['total_steps']
    result['steps_share_475_or_more'] = float(np.mean([r['learner_steps'] >= 475 for r in rows]) * 100)
    result['move_label_agreement_pct'] = 100 * sum(r['move_guidance']['agreements'] for r in rows) / sum(r['move_guidance']['labels'] for r in rows)
    result['combat_bomb_agreement_pct'] = 100 * sum(r['combat_bombs'] for r in rows) / sum(r['combat_labels'] for r in rows)
    result['teacher_move_entropy'] = sum(r['move_guidance']['entropy_sum'] for r in rows) / sum(r['move_guidance']['labels'] for r in rows)
    return result


def main():
    reg = verify_run(RUN)
    assert not (RUN / 'test_opened.json').exists()
    assert not any((RUN / 'test/tasks').glob('*.json'))
    candidates, snapshots = candidates_for_run(RUN, reg)
    tasks = validation_tasks(RUN)
    saved = _read_ledger(RUN / 'validation/tasks', tasks)
    assert len(saved) == len(tasks) == 1800
    selection = _selection(RUN, reg, candidates, snapshots, tasks)
    assert selection == read_json(RUN / 'selected.json')
    assert read_json(RUN / 'report.json') == _validation_report(selection, 2 * len(tasks))
    assert not selection['open_test']
    rows = _rows(saved)
    models = [candidates['S2'], *[d for arm in ('control', 'pool') for d in candidates[arm] if d['episode'] > 0]]
    by_model = {d['id']: [r for r in rows if r['learner_sha256'] == d['sha256']] for d in models}
    matchup, corners, checkpoints = [], [], []
    for desc in models:
        own = by_model[desc['id']]
        scores = {}
        for opp in (*CNN, 'RB'):
            match = [r for r in own if r['opponent_id'] == opp]
            m = summarize(match)
            scores[opp] = m
            matchup.append(dict(model=desc['id'], episode=desc['episode'], opponent=opp,
                games=m['games'], **{k: m[k] for k in ('active_win_rate', 'win_rate', 'draw_rate',
                    'suicide_rate', 'sudden_death_win_rate', 'sudden_death_death_rate',
                    'passive_win_rate', 'avg_kills', 'avg_bombs', 'avg_steps')}))
            for corner, v in m['by_corner'].items():
                corners.append(dict(model=desc['id'], opponent=opp, corner=int(corner), **v))
        checkpoints.append(dict(model=desc['id'], episode=desc['episode'],
            cnn_active=float(np.mean([scores[o]['active_win_rate'] for o in CNN])),
            cnn_wr=float(np.mean([scores[o]['win_rate'] for o in CNN])),
            cnn_draw=float(np.mean([scores[o]['draw_rate'] for o in CNN])),
            cnn_steps=float(np.mean([scores[o]['avg_steps'] for o in CNN])),
            rb_active=scores['RB']['active_win_rate'], rb_wr=scores['RB']['win_rate'],
            rb_draw=scores['RB']['draw_rate'], rb_sd_win=scores['RB']['sudden_death_win_rate'],
            rb_sd_death=scores['RB']['sudden_death_death_rate'],
            rb_suicide=scores['RB']['suicide_rate'], rb_bombs=scores['RB']['avg_bombs'],
            rb_steps=scores['RB']['avg_steps']))
    table('matchups.csv', matchup)
    table('corners.csv', corners)
    table('checkpoints.csv', checkpoints)
    print('Validated raw ledgers and checkpoint summaries.', flush=True)

    comparisons = {}
    for left, right in [('control-1000', 'S2'), ('control-2000', 'S2'), ('pool-2000', 'S2'),
                        ('pool-2000', 'control-2000'), ('pool-2000', 'control-1000')]:
        key = left + '_minus_' + right
        comparisons[key] = {}
        for group, opponents in [('cnn', CNN), ('RB', ('RB',))]:
            comparisons[key][group] = {metric: paired_macro_comparison(by_model[left], by_model[right], metric, opponents)
                for metric in ('active_win', 'win', 'draw', 'suicide', 'sudden_death_win', 'sudden_death_death')}
    dump('comparisons.json', comparisons)

    train_rows, audits, windows, train_opponents = {}, {}, [], []
    for arm in ('control', 'pool'):
        state = load_training_state(RUN / 'arms' / arm / 'training_state.pt', expected_format='bomberman_phase_d_training_v1')
        history = [json.loads(x) for x in (RUN / 'arms' / arm / 'metrics.jsonl').read_text().splitlines()]
        assert state['episode'] == 2000 and history == state['context']['metrics']
        assert [r['episode'] for r in history] == list(range(1, 2001))
        assert all(r['optimizer_steps'] == r['episode'] for r in history)
        schedule = build_train_schedule(arm, snapshots=state['context']['snapshots'])['episodes']
        for actual, expected in zip(history, schedule):
            assert all(actual[k] == v for k, v in expected.items()), (arm, actual['episode'])
        opt_steps = sorted({int(s['step'].item()) for s in state['optimizer']['state'].values()})
        assert opt_steps == [2000] and state['scheduler']['last_epoch'] == 2000
        opponent_audit = state['context']['opponent_audit']
        assert all(a['before_sha256'] == a['after_sha256'] and a['eval_mode'] and a['gradient_count'] == 0 for a in opponent_audit)
        numeric = ('reward', 'value_loss', 'policy_loss', 'combat_loss', 'navigation_loss', 'loss')
        assert all(np.isfinite(r[k]) for r in history for k in numeric)
        train_rows[arm] = history
        audits[arm] = dict(episode=state['episode'], optimizer_steps=opt_steps,
            scheduler_epoch=state['scheduler']['last_epoch'], opponent_audit_entries=len(opponent_audit),
            schedule_verified=True, metrics_equal_state=True, opponent_types=dict(collections.Counter(r['opponent_type'] for r in history)),
            recent_fallback_episodes=sum(r['recent_fallback'] for r in history), summary=training_summary(history))
        for start in range(1, 2001, 200):
            group = [r for r in history if start <= r['episode'] < start + 200]
            windows.append(dict(arm=arm, start=start, end=start+199, **training_summary(group)))
        for opp in sorted({r['opponent_id'] for r in history}):
            group = [r for r in history if r['opponent_id'] == opp]
            train_opponents.append(dict(arm=arm, opponent=opp, **training_summary(group)))
    assert train_rows['control'][:2] == train_rows['pool'][:2]
    table('training_windows.csv', windows)
    table('training_opponents.csv', train_opponents)
    dump('training_audit.json', audits)
    print('Verified training schedule, state histories, optimizer and frozen opponents.', flush=True)

    # Only already-collected SD8A TRAIN-partition observations. No environments,
    # held-out states, search, backward pass, optimizer step or checkpoint writes.
    old = REPO / 'runs/sd8a-20261020'
    old_reg = read_json(old / 'registration.json')
    files = sorted((old / 'collection').glob('*.npz'))
    seed_lists = {kind: [] for kind in ('decision', 'anchor')}
    hashes = {}
    for path in files:
        hashes[str(path.relative_to(REPO))] = sha(path)
        with np.load(path, allow_pickle=False) as data:
            for kind in seed_lists:
                seed_lists[kind].append(data[kind + '_seed'])
    all_seeds = np.unique(np.concatenate([a for kind in seed_lists for a in seed_lists[kind]]))
    n_held = max(1, int(round(len(all_seeds) * old_reg['config']['heldout_fraction'])))
    held = set(np.random.default_rng(530).permutation(all_seeds)[:n_held].tolist())
    rng = np.random.default_rng(20261023)
    samples, sample_meta = {}, {}
    for kind in ('decision', 'anchor'):
        eligible = [(fi, int(ri)) for fi, seeds in enumerate(seed_lists[kind])
                    for ri in np.flatnonzero(~np.isin(seeds, list(held)))]
        picks = sorted(eligible[i] for i in rng.choice(len(eligible), min(1024, len(eligible)), replace=False))
        wanted = collections.defaultdict(list)
        for fi, ri in picks:
            wanted[fi].append(ri)
        chunks = collections.defaultdict(list)
        fields = ('obs', 'mask', 'seed', 'scores') if kind == 'decision' else ('obs', 'mask', 'seed')
        for fi, indices in wanted.items():
            with np.load(files[fi], allow_pickle=False) as data:
                for field in fields:
                    chunks[field].append(data[kind + '_' + field][indices])
        sample = {k: np.concatenate(v) for k, v in chunks.items()}
        assert not held.intersection(sample['seed'].tolist())
        if kind == 'decision':
            scores = np.where(sample['mask'] & np.isfinite(sample['scores']), sample['scores'], -np.inf)
            best = scores.max(1, keepdims=True)
            valid = ~np.all((scores == best) | ~sample['mask'], axis=1)
            sample = {k: v[valid] for k, v in sample.items()}
            scores = scores[valid]
            winners = sample['mask'] & (scores == scores.max(1, keepdims=True))
            sample['best'] = winners
            sample['target'] = winners.astype(np.float32) / winners.sum(1, keepdims=True)
        samples[kind] = sample
        sample_meta[kind] = dict(samples=len(sample['seed']), boards=len(np.unique(sample['seed'])),
            seed_min=int(sample['seed'].min()), seed_max=int(sample['seed'].max()),
            identity_digest=canonical_digest([list(x) for x in picks]),
            selected_rows=[dict(file=str(files[fi].relative_to(REPO)), row=ri) for fi, ri in picks])
    dump('offline_sample_manifest.json', dict(seed=20261023, train_only=True,
        original_heldout_seed=530, excluded_heldout_board_ids=sorted(held), input_sha256=hashes, samples=sample_meta))

    def predict(desc, sample):
        net = load_registered_model(desc['checkpoint'], desc['id'], device='cpu')
        predictions, values = [], []
        with torch.inference_mode():
            for start in range(0, len(sample['obs']), 128):
                logits, value = net(torch.from_numpy(sample['obs'][start:start+128]).float())
                mask = torch.from_numpy(sample['mask'][start:start+128]).bool()
                predictions.append(torch.log_softmax(logits.masked_fill(~mask, -1e9), -1).numpy())
                values.append(value.numpy())
        return np.concatenate(predictions), np.concatenate(values)

    reference = {kind: predict(models[0], sample)[0] for kind, sample in samples.items()}
    retention = []
    for desc in models:
        result = dict(model=desc['id'], episode=desc['episode'])
        for kind, sample in samples.items():
            log_p, values = predict(desc, sample)
            ref = reference[kind]
            probs, ref_probs = np.exp(log_p), np.exp(ref)
            choice = log_p.argmax(1)
            result[kind + '_samples'] = len(log_p)
            result[kind + '_kl_S2_to_model'] = float(np.mean(np.sum(ref_probs * (ref - log_p), axis=1)))
            result[kind + '_changed_argmax_pct'] = float(np.mean(choice != ref.argmax(1)) * 100)
            result[kind + '_bomb_probability_pct'] = float(np.mean(probs[:, ACTION_BOMB]) * 100)
            result[kind + '_value_mean'] = float(values.mean())
            if kind == 'decision':
                result['search_best_argmax_pct'] = float(np.mean(sample['best'][np.arange(len(choice)), choice]) * 100)
                result['search_best_probability_pct'] = float(np.mean(np.sum(probs * sample['best'], axis=1)) * 100)
                result['search_target_ce'] = float(-np.mean(np.sum(sample['target'] * log_p, axis=1)))
        assert all(np.isfinite(v) for v in result.values() if isinstance(v, float))
        retention.append(result)
        print('Offline train-state retention:', desc['id'], flush=True)
    table('offline_retention.csv', retention)

    initial = torch.load(models[0]['checkpoint'], map_location='cpu', weights_only=True)
    g = torch.load(reg['checkpoints']['G']['checkpoint'], map_location='cpu', weights_only=True)
    critic_equal = all(torch.equal(initial[k], g[k]) for k in initial if k.startswith('critic_head.'))
    drifts = []
    for desc in models:
        weights = torch.load(desc['checkpoint'], map_location='cpu', weights_only=True)
        row = dict(model=desc['id'])
        for prefix in ('conv.', 'trunk_fc.', 'actor_head.', 'critic_head.'):
            keys = [k for k in initial if k.startswith(prefix)]
            num = sum(float(torch.sum((weights[k].double()-initial[k].double())**2)) for k in keys)
            den = sum(float(torch.sum(initial[k].double()**2)) for k in keys)
            row[prefix.rstrip('.') + '_relative_l2'] = float(np.sqrt(num / den))
        drifts.append(row)
    table('parameter_drift.csv', drifts)
    summary = dict(status='POST_HOC_VALIDATION_DIAGNOSTIC', registration_digest=reg['registration_digest'],
        validation_tasks=len(saved), validation_games=len(rows), selected_models={k:v['id'] for k,v in selection['models'].items()},
        test_opened=False, new_games=0, updates=0, seed_ranges_opened=[],
        baseline_critic_head_identical_to_G=critic_equal, checkpoints=checkpoints,
        training=audits, offline_retention=retention,
        limitations=['One training seed; only 40 validation boards per opponent.',
            'Post-hoc multiple diagnostic comparisons; confidence intervals are unadjusted and exploratory.',
            'Offline SD8A training states are reused diagnostics, not independent validation or game win estimates.',
            'Loss magnitudes do not establish relative gradient magnitudes or a causal mechanism.',
            'CPU validation versus CUDA training and data distribution differences are confounded.',
            'Critic mismatch is a hypothesis; critic targets/calibration were not directly measured.'])
    dump('summary.json', summary)
    print(json.dumps(dict(validation_games=len(rows), critic_head_identical_to_G=critic_equal,
        retention=[{k:r[k] for k in ('model','search_best_argmax_pct','search_best_probability_pct','anchor_kl_S2_to_model')} for r in retention],
        pool_steps=audits['pool']['summary']['total_steps'], control_steps=audits['control']['summary']['total_steps']), indent=2), flush=True)


if __name__ == '__main__':
    main()
