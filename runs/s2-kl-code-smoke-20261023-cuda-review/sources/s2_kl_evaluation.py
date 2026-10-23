"""Forward-only retention, paired board statistics and fixed-endpoint decisions."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from phase_d_s2_evaluation import RATES, load_registered_model, summarize
from phase_d_s2_protocol import validate_task_payload
from s2_kl_protocol import (SAMPLE_REL, SAMPLE_SHA, S2_SHA, _inputs, canonical_digest,
    configure_runtime, input_path, read_json, sha)
from s2_kl_training import model_manifest

CONDITIONS = ('DRIFT_REPRODUCED', 'DECISION_RETAINED', 'PROBABILITY_RETAINED',
    'ANCHOR_BENEFIT', 'RB_ACTIVE_RETAINED', 'RB_SUICIDE_RETAINED')


def decision_targets(scores, mask):
    scores, mask = np.asarray(scores), np.asarray(mask, dtype=bool)
    if scores.ndim != 2 or scores.shape != mask.shape:
        raise ValueError('Decision score/mask shape mismatch')
    legal = mask & np.isfinite(scores)
    safe = np.where(legal, scores, -np.inf)
    best = safe.max(axis=1, keepdims=True)
    winners = legal & (safe == best)
    valid = legal.any(axis=1) & (legal & ~winners).any(axis=1)
    targets = np.divide(winners.astype(np.float32), winners.sum(axis=1, keepdims=True),
        out=np.zeros(scores.shape, dtype=np.float32), where=winners.sum(axis=1, keepdims=True) > 0)
    targets[~valid] = 0
    return targets, valid


def load_samples(reg):
    checkpoints, offline = _inputs(reg['input_root'])
    if checkpoints != reg['checkpoints'] or offline != reg['offline_manifest']:
        raise ValueError('Offline inputs drift')
    if sha(Path(reg['output']) / 'offline_sample_manifest.json') != SAMPLE_SHA:
        raise ValueError('Copied sample manifest drift')
    manifest = read_json(input_path(reg['input_root'], SAMPLE_REL))
    if (manifest['train_only'] is not True or manifest['seed'] != 20261023 or
            manifest['original_heldout_seed'] != 530):
        raise ValueError('Locked TRAIN split identity drift')
    files = sorted(manifest['input_sha256'])
    selected = {kind: defaultdict(list) for kind in ('decision', 'anchor')}
    for kind in selected:
        info = manifest['samples'][kind]
        if info['samples'] != 1024 or len(info['selected_rows']) != 1024:
            raise ValueError('Locked sample count mismatch')
        identities = []
        for row in info['selected_rows']:
            if row['file'] not in manifest['input_sha256'] or type(row['row']) is not int or row['row'] < 0:
                raise ValueError('Invalid locked shard/row')
            selected[kind][row['file']].append(row['row'])
            identities.append([files.index(row['file']), row['row']])
        if canonical_digest(identities) != info['identity_digest'] or len({tuple(v) for v in identities}) != 1024:
            raise ValueError('Locked sample identity mismatch')
    chunks = {kind: defaultdict(list) for kind in selected}
    all_seeds = []
    for relative in files:
        path = input_path(reg['input_root'], relative)
        if sha(path) != manifest['input_sha256'][relative]:
            raise ValueError('Collection shard drift')
        with np.load(path, allow_pickle=False) as data:
            for kind in selected:
                seeds = data[kind + '_seed']
                all_seeds.append(seeds)
                indices = selected[kind][relative]
                if not indices:
                    continue
                if max(indices) >= len(seeds):
                    raise ValueError('Locked row outside shard')
                obs, mask = data[kind + '_obs'][indices], data[kind + '_mask'][indices]
                if (obs.shape != (len(indices), 13, 9, 9) or mask.shape != (len(indices), 6) or
                        mask.dtype != np.bool_ or not np.isfinite(obs).all() or not mask.any(axis=1).all()):
                    raise ValueError('Invalid locked observation/mask shape')
                chunks[kind]['obs'].append(obs)
                chunks[kind]['mask'].append(mask)
                chunks[kind]['boards'].append(seeds[indices])
                chunks[kind]['ids'].extend(str(relative).replace('\\', '/') + ':' + str(i) for i in indices)
                if kind == 'decision':
                    scores = data['decision_scores'][indices]
                    if scores.shape != mask.shape:
                        raise ValueError('Invalid decision score shape')
                    chunks[kind]['scores'].append(scores)
    seeds = np.unique(np.concatenate(all_seeds))
    held = sorted(np.random.default_rng(530).permutation(seeds)[:max(1, int(round(len(seeds) * .1)))].tolist())
    if held != manifest['excluded_heldout_board_ids'] or len(held) != 150:
        raise ValueError('Original held-out split drift')
    result = {}
    for kind, data in chunks.items():
        result[kind] = {key: (value if key == 'ids' else np.concatenate(value)) for key, value in data.items()}
        sample = result[kind]
        info = manifest['samples'][kind]
        if (len(sample['ids']) != 1024 or len(set(sample['ids'])) != 1024 or
                len(np.unique(sample['boards'])) != info['boards'] or set(sample['boards']) & set(held)):
            raise ValueError('Locked sample count/board/held-out mismatch')
    targets, valid = decision_targets(result['decision']['scores'], result['decision']['mask'])
    if not valid.all():
        raise ValueError('Locked decision rows contain noninformative labels')
    result['decision']['targets'] = targets
    return result


def diagnostic_summary(raw):
    decision = raw['decision']
    result = dict(R=100 * float(np.mean([r['hit'] for r in decision])),
        Q=100 * float(np.mean([r['mass'] for r in decision])))
    for kind, rows in raw.items():
        result[kind] = dict(samples=len(rows),
            kl_S2_to_model=float(np.mean([r['kl_s2_to_model'] for r in rows])),
            bomb_probability_pct=100 * float(np.mean([r['bomb_probability'] for r in rows])),
            value_mean=float(np.mean([r['value'] for r in rows])))
        if all('changed_argmax' in r for r in rows):
            result[kind]['changed_argmax_pct'] = 100 * float(np.mean([r['changed_argmax'] for r in rows]))
    result['decision']['soft_target_ce'] = float(np.mean([r['ce'] for r in decision]))
    return result


def diagnose_model(model, reg):
    import torch
    import torch.nn.functional as F
    from bomberman_env import ACTION_BOMB
    if sha(model['checkpoint']) != model['sha256']:
        raise ValueError('Diagnostic model hash drift')
    configure_runtime('cpu')
    samples = load_samples(reg)
    net = load_registered_model(model['checkpoint'], 'S2', 'cpu')
    reference = load_registered_model(reg['checkpoints']['S2']['checkpoint'], 'S2', 'cpu')
    raw = {kind: [] for kind in samples}
    with torch.inference_mode():
        for kind, sample in samples.items():
            for start in range(0, 1024, 128):
                obs = torch.as_tensor(sample['obs'][start:start + 128], dtype=torch.float32)
                mask = torch.as_tensor(sample['mask'][start:start + 128], dtype=torch.bool)
                logits, values = net(obs)
                ref_logits, _ = reference(obs)
                lp = F.log_softmax(logits.masked_fill(~mask, -1e9), -1)
                lref = F.log_softmax(ref_logits.masked_fill(~mask, -1e9), -1)
                probabilities, ref_probabilities = lp.exp(), lref.exp()
                choices, ref_choices = lp.argmax(-1), lref.argmax(-1)
                kl = (ref_probabilities * (lref - lp)).sum(-1)
                for offset in range(len(obs)):
                    index = start + offset
                    row = dict(id=sample['ids'][index], board=int(sample['boards'][index]),
                        kl_s2_to_model=float(kl[offset]), bomb_probability=float(probabilities[offset, ACTION_BOMB]),
                        value=float(values[offset]), changed_argmax=int(choices[offset] != ref_choices[offset]))
                    if kind == 'decision':
                        target = torch.as_tensor(sample['targets'][index])
                        good = target > 0
                        row.update(hit=int(good[choices[offset]]),
                            mass=min(1., max(0., float(probabilities[offset, good].sum()))),
                            ce=float(-(target * lp[offset]).sum()))
                    if any(not math.isfinite(v) for k, v in row.items() if k != 'id'):
                        raise ValueError('Nonfinite diagnostic observation')
                    raw[kind].append(row)
    result = dict(format='s2-kl-diagnostics-v1', model_sha256=model['sha256'],
        registration_digest=reg['registration_digest'], sample_manifest_sha256=SAMPLE_SHA,
        raw=raw, summary=diagnostic_summary(raw))
    result['diagnostic_digest'] = canonical_digest(result)
    return result


def validation_tasks(reg, models):
    tasks = []
    unique = {m['sha256']: m for m in reversed(models)}
    span = reg['seed_manifest']['ranges']['validation']
    opponent = dict(id='RB', arm=None, episode=None, checkpoint=None,
        sha256=reg['sources']['rule_based_bot.py'])
    for model in reversed(list(unique.values())):
        for board in range(span['start'], span['end_inclusive'] + 1):
            identity = dict(stage='validation', registration_digest=reg['registration_digest'],
                learner_sha256=model['sha256'], opponent_sha256=opponent['sha256'], board=board)
            tasks.append(dict(id=canonical_digest(identity), stage='validation', learner=deepcopy(model),
                opponent=deepcopy(opponent), board=board, corners=[0, 1],
                max_steps=reg['config']['train']['max_steps'], registration_digest=reg['registration_digest'],
                selection_digest=None))
    return tasks


def cluster_comparison(left, right, *, resamples=10000, seed=20261023):
    if type(resamples) is not int or resamples < 1 or type(seed) is not int:
        raise ValueError('Invalid bootstrap parameters')
    def indexed(rows):
        result = {}
        for row in rows:
            if (not isinstance(row.get('id'), str) or row['id'] in result or type(row.get('board')) is not int or
                    not isinstance(row.get('value'), (float, int)) or
                    not math.isfinite(row['value']) or not 0 <= row['value'] <= 1):
                raise ValueError('Invalid paired observation')
            result[row['id']] = row
        if not result:
            raise ValueError('Empty paired observations')
        return result
    a, b = indexed(left), indexed(right)
    if set(a) != set(b) or any(a[k]['board'] != b[k]['board'] for k in a):
        raise ValueError('Paired ID/board mismatch')
    clusters = defaultdict(list)
    for key in sorted(a):
        clusters[a[key]['board']].append(a[key]['value'] - b[key]['value'])
    sums = np.array([sum(clusters[k]) for k in sorted(clusters)], dtype=np.float64)
    counts = np.array([len(clusters[k]) for k in sorted(clusters)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, resamples, 256):
        chosen = rng.integers(0, len(counts), size=(min(256, resamples - start), len(counts)))
        draws.append(100 * sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1))
    return dict(difference_pp=float(100 * sums.sum() / counts.sum()),
        ci95_pp=np.quantile(np.concatenate(draws), [.025, .975], method='linear').tolist(),
        board_clusters=len(counts), samples=len(a), resamples=resamples, bootstrap_seed=seed)


def decide_gate(baseline, control, anchor, comparisons, *, complete, smoke=False):
    for metrics in (baseline, control, anchor):
        if any(not isinstance(metrics.get(k), (int, float)) or not math.isfinite(metrics[k]) or
                not 0 <= metrics[k] <= 100 for k in ('R', 'Q', 'AW', 'SU')):
            raise ValueError('Invalid endpoint metrics')
    comparison = comparisons['anchor_minus_control']['R']
    if (len(comparison['ci95_pp']) != 2 or any(not math.isfinite(v) for v in
            [comparison['difference_pp'], *comparison['ci95_pp']]) or
            comparison['ci95_pp'][0] > comparison['ci95_pp'][1]):
        raise ValueError('Invalid retention confidence interval')
    conditions = dict(
        DRIFT_REPRODUCED=control['R'] <= baseline['R'] - 10,
        DECISION_RETAINED=anchor['R'] >= baseline['R'] - 5,
        PROBABILITY_RETAINED=anchor['Q'] >= baseline['Q'] - 5,
        ANCHOR_BENEFIT=anchor['R'] - control['R'] >= 10 and comparison['ci95_pp'][0] > 0,
        RB_ACTIVE_RETAINED=anchor['AW'] >= baseline['AW'] - 2,
        RB_SUICIDE_RETAINED=anchor['SU'] <= baseline['SU'] + 1)
    if not complete:
        status = 'INCOMPLETE'
    elif smoke:
        status = 'SMOKE_ONLY'
    elif not conditions['DRIFT_REPRODUCED']:
        status = 'INCONCLUSIVE_NO_DRIFT'
    elif not (conditions['RB_ACTIVE_RETAINED'] and conditions['RB_SUICIDE_RETAINED']):
        status = 'RB_RETENTION_FAIL'
    elif not (conditions['DECISION_RETAINED'] and conditions['PROBABILITY_RETAINED']):
        status = 'RETENTION_NOT_PRESERVED'
    elif not conditions['ANCHOR_BENEFIT']:
        status = 'NO_CLEAR_ANCHOR_BENEFIT'
    else:
        status = 'RETENTION_PILOT_PASS'
    return dict(status=status, complete=bool(complete), conditions=conditions,
        **{'pass': status == 'RETENTION_PILOT_PASS'})


def validate_diagnostic(payload, model, reg, samples):
    if (payload.get('format') != 's2-kl-diagnostics-v1' or payload.get('model_sha256') != model['sha256'] or
            payload.get('registration_digest') != reg['registration_digest'] or
            payload.get('sample_manifest_sha256') != SAMPLE_SHA or
            canonical_digest(payload, 'diagnostic_digest') != payload.get('diagnostic_digest')):
        raise ValueError('Diagnostic identity/digest drift')
    for kind in ('decision', 'anchor'):
        rows = payload.get('raw', {}).get(kind, [])
        sample = samples[kind]
        if len(rows) != 1024 or [r.get('id') for r in rows] != sample['ids'] or [r.get('board') for r in rows] != sample['boards'].tolist():
            raise ValueError('Diagnostic locked row identity/count drift')
        required = ('kl_s2_to_model', 'bomb_probability', 'value') + (('hit', 'mass', 'ce') if kind == 'decision' else ())
        for row in rows:
            if any(not isinstance(row.get(k), (float, int)) or not math.isfinite(row[k]) for k in required):
                raise ValueError('Nonfinite diagnostic values')
            if not 0 <= row['bomb_probability'] <= 1 or (kind == 'decision' and
                    (row['hit'] not in (0, 1) or not 0 <= row['mass'] <= 1)):
                raise ValueError('Invalid diagnostic probability/hit')
    calculated = diagnostic_summary(payload['raw'])
    def check(actual, expected):
        if isinstance(actual, dict):
            if not isinstance(expected, dict) or any(k not in expected for k in actual):
                raise ValueError('Diagnostic summary schema drift')
            for k, value in actual.items():
                check(value, expected[k])
        elif not isinstance(actual, (float, int)) or not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-12):
            raise ValueError('Diagnostic summary/raw mismatch')
    if not {'R', 'Q'} <= payload.get('summary', {}).keys():
        raise ValueError('Missing retention summary')
    check(payload['summary'], calculated)
    return calculated


def build_report(reg, models, diagnostics, rb_rows):
    rb_rows = sorted(rb_rows, key=lambda r: (r['learner_sha256'], r['board'], r['corner']))
    descriptors = list(models)
    by_sha = {}
    for model in descriptors:
        if model['sha256'] in by_sha or sha(model['checkpoint']) != model['sha256']:
            raise ValueError('Duplicate/drifted model descriptor')
        by_sha[model['sha256']] = model
    manifest = model_manifest(reg, descriptors) if descriptors else None
    # Evaluation deduplicates SHA, but the committed inventory retains every
    # candidate/endpoint alias, including two arms with identical checkpoint bytes.
    manifest_path = Path(reg['output']) / 'model_manifest.json'
    if descriptors and manifest_path.is_file():
        manifest = read_json(manifest_path)
        if (manifest.get('registration_digest') != reg['registration_digest'] or
                canonical_digest(manifest, 'model_manifest_digest') != manifest.get('model_manifest_digest') or
                manifest.get('models') != descriptors):
            raise ValueError('Model manifest identity/inventory drift')
    tasks = validation_tasks(reg, descriptors)
    expected = {(t['learner']['sha256'], t['board']): t for t in tasks}
    groups = defaultdict(list)
    seen = set()
    for row in rb_rows:
        key = (row['learner_sha256'], row['board'])
        identity = (*key, row['corner'])
        if key not in expected or row['corner'] not in (0, 1) or identity in seen:
            raise ValueError('Unexpected/duplicate RB board corner')
        seen.add(identity)
        task = expected[key]
        if (row['opponent_id'] != 'RB' or row['opponent_sha256'] != task['opponent']['sha256'] or
                any(row.get(k) not in (0, 1) for k in RATES) or
                any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k]) or row[k] < 0 for k in ('kills', 'bombs', 'steps')) or
                row['steps'] > task['max_steps'] or row['win'] + row['draw'] > 1 or
                row['win'] != row['active_win'] + row['passive_win'] + row['sudden_death_win']):
            raise ValueError('Invalid RB row')
        groups[key].append(row)
    for key, rows in groups.items():
        if len(rows) == 2:
            task = expected[key]
            validate_task_payload(task, dict(format='d-s2-board-result-v1', id=task['id'],
                registration_digest=reg['registration_digest'], selection_digest=None, records=rows))
    samples = load_samples(reg) if diagnostics else None
    if set(diagnostics) - set(by_sha):
        raise ValueError('Unexpected diagnostic model')
    summary = {}
    for digest, model in by_sha.items():
        entry = dict(model=model)
        if digest in diagnostics:
            entry['offline'] = validate_diagnostic(diagnostics[digest], model, reg, samples)
        own = [r for r in rb_rows if r['learner_sha256'] == digest]
        if own:
            entry['RB'] = summarize(own)
        summary[model['id']] = entry
    endpoint = reg['config']['endpoint_episode']
    inventory_complete = (manifest is not None and len(manifest['endpoints']) == 3 and
        all(manifest['aliases'].get(f'{arm}-{ep}') in by_sha
            for arm in ('control', 'anchor') for ep in reg['config']['checkpoint_episodes']))
    complete = bool(inventory_complete and set(diagnostics) == set(by_sha) and
        len(rb_rows) == 2 * len(tasks) and all(len(groups[key]) == 2 for key in expected))
    comparisons = {}
    endpoints = manifest['endpoints'] if manifest else {}
    if len(endpoints) == 3:
        for name, left, right in [('control_minus_baseline', 'control500', 'baseline'),
                                  ('anchor_minus_baseline', 'anchor500', 'baseline'),
                                  ('anchor_minus_control', 'anchor500', 'control500')]:
            a, b = endpoints[left]['sha256'], endpoints[right]['sha256']
            compared = {}
            if a in diagnostics and b in diagnostics:
                for label, field in [('R', 'hit'), ('Q', 'mass')]:
                    pairs = [[dict(id=r['id'], board=r['board'], value=r[field]) for r in diagnostics[d]['raw']['decision']] for d in (a, b)]
                    compared[label] = cluster_comparison(*pairs)
            a_rows = [r for r in rb_rows if r['learner_sha256'] == a]
            b_rows = [r for r in rb_rows if r['learner_sha256'] == b]
            span = reg['seed_manifest']['ranges']['validation']
            games = 2 * (span['end_inclusive'] - span['start'] + 1)
            if len(a_rows) == len(b_rows) == games:
                for field in RATES:
                    pairs = [[dict(id=f"{r['board']}:{r['corner']}", board=r['board'], value=r[field]) for r in group] for group in (a_rows, b_rows)]
                    compared['RB_' + field] = cluster_comparison(*pairs)
            comparisons[name] = compared
    conditions = {k: False for k in CONDITIONS}
    gate = dict(status='INCOMPLETE', complete=False, conditions=conditions, **{'pass': False})
    if len(endpoints) == 3 and all('offline' in summary[by_sha[d['sha256']]['id']] and
            'RB' in summary[by_sha[d['sha256']]['id']] for d in endpoints.values()) and 'R' in comparisons.get('anchor_minus_control', {}):
        values = []
        for key in ('baseline', 'control500', 'anchor500'):
            entry = summary[by_sha[endpoints[key]['sha256']]['id']]
            values.append(dict(R=entry['offline']['R'], Q=entry['offline']['Q'],
                AW=entry['RB']['active_win_rate'], SU=entry['RB']['suicide_rate']))
        gate = decide_gate(*values, comparisons, complete=complete, smoke=reg['mode'] == 'smoke')
    return dict(**gate, mode=reg['mode'], endpoints=endpoints,
        counts=dict(models=len(by_sha), diagnostics=len(diagnostics), expected_diagnostics=len(by_sha),
            games=len(rb_rows), expected_games=2 * len(tasks),
            decision_states=sum(len(d['raw']['decision']) for d in diagnostics.values()),
            anchor_states=sum(len(d['raw']['anchor']) for d in diagnostics.values())),
        summary=summary, comparisons=comparisons, registration_digest=reg['registration_digest'],
        model_manifest_digest=manifest['model_manifest_digest'] if manifest else None,
        limitations=['One training seed; reused SD8A TRAIN observations measure retention only.',
            'Only fixed endpoint ' + str(endpoint) + ' decides the pilot; intermediate checkpoints are diagnostic.',
            'No held-out stage and no automatic replacement of S2 or Phase D qualification.',
            'Smoke with 20-step episodes does not establish full-run throughput or model quality.'] if reg['mode'] == 'smoke' else
            ['One training seed; reused SD8A TRAIN observations measure retention only.',
             'Only fixed endpoint 500 decides the pilot; intermediate checkpoints are diagnostic.',
             'No held-out stage and no automatic replacement of S2 or Phase D qualification.'])
