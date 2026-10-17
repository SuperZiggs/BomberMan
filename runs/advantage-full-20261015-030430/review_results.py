"""Recompute T4-A1 provenance, diagnostics, validation, skill tapes and paired test gates."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1]).resolve()
import numpy as np
import torch
from audit_t2 import digest, dump
from audit_t3_approach import replay_labels
from benchmark_t3 import scenarios, summarize_games
from pilot_two_bomb import paired_active_bootstrap
from run_t3_curriculum import model_digest
from train_actor_critic import ActorCriticNet, smart_load_weights

torch.set_num_threads(1)
config = json.loads((OUT / 'config.json').read_text())
summary = json.loads((OUT / 'summary.json').read_text())
assert summary['status'] == 'complete' and summary['full_protocol'] and summary['hashes_unchanged']
assert config['train_seed'] == 2026101502 and config['episodes_per_arm'] == 200
assert config['evaluation']['seed'] == 34000 and config['evaluation']['games_per_model'] == 400
for name, expected in config['source_sha256'].items():
    assert digest(ROOT / name) == digest(OUT / 'source' / name) == expected
assert digest(Path(config['warm_start'])) == config['warm_start_sha256']
archive = Path(str(OUT) + '-send.zip')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for entry in z.infolist():
        if not entry.is_dir():
            assert z.read(entry) == (OUT / entry.filename).read_bytes(), entry.filename
    archive_files = sum(not e.is_dir() for e in z.infolist())

registry = {c['id']: c for c in scenarios(('validation',))}
arm_tables, skill_tables, replay_frames, configs = {}, {}, 0, {}
for arm, normalize in config['arms'].items():
    folder = OUT / arm
    cfg = json.loads((folder / 'config.json').read_text())
    configs[arm] = cfg
    assert cfg['attack_teacher'] == 'local' and cfg['normalize_advantage'] == normalize
    assert cfg['advantage_diagnostics'] and cfg['resume_from'] == config['warm_start']
    assert (cfg['episodes'], cfg['seed'], cfg['lr'], cfg['max_steps']) == (200, 2026101502, 1e-4, 500)
    rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    assert [r['episode'] for r in rows] == list(range(1, 201))
    assert [r['optimizer_steps'] for r in rows] == list(range(1, 201))
    assert all(r['learning_aid'] == (r['episode'] - 1) % 2 for r in rows)
    for row in rows:
        d = row['advantage']
        assert row['attack_teacher'] == 'local' and 0 < row['learner_steps'] <= 500
        assert d['normalize_enabled'] == normalize and d['samples'] == row['learner_steps']
        assert d['normalization_skipped'] == (normalize and (d['samples'] < 2 or d['raw']['std'] < 1e-8))
        if normalize and not d['normalization_skipped']:
            assert abs(d['policy']['mean']) < 1e-4 and abs(d['policy']['std'] - 1) < 1e-5
        else:
            assert d['raw'] == d['policy']
        assert d['grad_norm_before_clip'] >= 0 and math.isfinite(d['grad_norm_before_clip'])
        assert d['gradient_clipped'] == (d['grad_norm_before_clip'] > 5)
        assert math.isclose(row['value_loss'], d['raw']['std'] ** 2 + d['raw']['mean'] ** 2,
                            rel_tol=1e-5, abs_tol=1e-5)
        assert sum(s['samples'] for s in d['entropy_by_mask_size'].values()) == d['samples']
        for size, entropy in d['entropy_by_mask_size'].items():
            assert 1 <= int(size) <= 6 and 0 <= entropy['mean'] <= np.log(int(size)) + 1e-5
        for field in ('policy_loss', 'value_loss', 'escape_loss', 'combat_loss', 'navigation_loss', 'loss', 'reward'):
            assert math.isfinite(row[field])
        assert row['move_guidance']['queries'] == row['move_guidance']['accepted'] == 0
    vals = [json.loads(line) for line in (folder / 'validation_metrics.jsonl').read_text().splitlines()]
    assert [r['episode'] for r in vals] == [0, 100, 200]
    assert all((v['num_episodes'], v['seed'], v['device'], v['max_steps']) == (100, 21000, 'cpu', 500) for v in vals)
    guard, score, best = vals[0]['suicide_rate'], (-float('inf'),), None
    for v in vals:
        candidate = (v['active_win_rate'], -v['suicide_rate'], -v['draw_rate'], v['avg_kills'], v['win_rate'])
        selected = v['suicide_rate'] <= guard and candidate > score
        assert v['selected'] == selected and v['suicide_limit'] == guard
        if selected:
            score, best = candidate, v['episode']
    assert json.loads((folder / 'best_validation.json').read_text())['episode'] == best
    probes = [json.loads(line) for line in (folder / 'skill_metrics.jsonl').read_text().splitlines()]
    assert [p['episode'] for p in probes] == [0, 100, 200]
    games = [json.loads(line) for line in gzip.open(folder / 'skill_games.jsonl.gz', 'rt')]
    assert len(games) == 192 and all(g['split'] == 'validation' and g['opponent'] == 'static' for g in games)
    for game in games:
        replay_frames += len(replay_labels(game, registry[game['scenario_id']]))
    per_probe = {}
    for probe in probes:
        selected_games = [g for g in games if g['episode'] == probe['episode']]
        assert len(selected_games) == 64
        recomputed = {g: summarize_games([r for r in selected_games if r['group'] == g]) for g in ('escape', 'approach')}
        assert recomputed == probe['groups']
        per_probe[str(probe['episode'])] = recomputed
    for checkpoint, episode in (('best', best), ('final', 200)):
        net = ActorCriticNet(board_size=9)
        path = folder / f'{checkpoint}_phaseC.pt'
        smart_load_weights(net, str(path), 'cpu')
        assert model_digest(net) == next(p['model_state_sha256'] for p in probes if p['episode'] == episode)
        assert digest(path) == summary['arms'][arm]['checkpoints'][checkpoint]['sha256']
    skill_tables[arm] = per_probe
    grad_norms = [r['advantage']['grad_norm_before_clip'] for r in rows]
    explained = [r['advantage']['explained_variance'] for r in rows if r['advantage']['explained_variance'] is not None]
    arm_tables[arm] = {'best_episode': best, 'learner_steps': sum(r['learner_steps'] for r in rows),
        'training_suicides': sum(r['suicide'] for r in rows), 'training_active_wins': sum(r['active_win'] for r in rows),
        'clipped_updates': sum(r['advantage']['gradient_clipped'] for r in rows),
        'normalization_skipped': sum(r['advantage']['normalization_skipped'] for r in rows),
        'grad_norm_mean': float(np.mean(grad_norms)), 'grad_norm_p95': float(np.percentile(grad_norms, 95)),
        'explained_variance_mean': float(np.mean(explained)) if explained else None,
        'advantage_raw_std_mean': float(np.mean([r['advantage']['raw']['std'] for r in rows])),
        'advantage_policy_std_mean': float(np.mean([r['advantage']['policy']['std'] for r in rows])),
        'entropy_mean_per_episode': float(np.mean([r['advantage']['entropy_mean'] for r in rows])),
        'validations': vals, 'seconds': summary['arms'][arm]['elapsed_seconds']}
assert {k: v for k, v in configs['control'].items() if k not in ('checkpoint_dir', 'normalize_advantage')} == {
    k: v for k, v in configs['normalized'].items() if k not in ('checkpoint_dir', 'normalize_advantage')}

models, tables = {}, {}
for name in ('original', 'control', 'normalized'):
    r = json.loads((OUT / f'evaluation_{name}_34000.json').read_text())
    assert (r['seed'], r['num_episodes'], r['num_agents'], r['max_steps']) == (34000, 400, 2, 500)
    assert [(g['seed'], g['aid']) for g in r['games']] == [(s, a) for s in range(34000, 34200) for a in (0, 1)]
    assert digest(Path(r['checkpoint'])) == r['checkpoint_sha256']
    for aid in (None, 0, 1):
        games = [g for g in r['games'] if aid is None or g['aid'] == aid]
        stats = r if aid is None else r['by_corner'][str(aid)]
        for field in ('win', 'active_win', 'suicide', 'draw'):
            assert abs(stats[f'{field}_rate'] - 100 * sum(g[field] for g in games) / len(games)) < 1e-9
        for field in ('kills', 'bombs', 'items', 'steps'):
            assert abs(stats[f'avg_{field}'] - sum(g[field] for g in games) / len(games)) < 1e-9
        assert abs(stats['passive_win_rate'] - (stats['win_rate'] - stats['active_win_rate'])) < 1e-9
    models[name] = r
    tables[name] = {'counts': {f: sum(g[f] for g in r['games']) for f in ('win', 'active_win', 'suicide', 'draw')},
                    **{k: v for k, v in r.items() if k != 'games'}}
comparisons = {name: paired_active_bootstrap(models['normalized'], models[name]) for name in ('original', 'control')}
assert comparisons == summary['comparisons']
gates = {'gain': all(c['difference_pp'] >= 5 for c in comparisons.values()),
         'ci': all(c['ci95_pp'][0] > 0 for c in comparisons.values()),
         'suicide': all(models['normalized']['suicide_rate'] <= models[name]['suicide_rate'] for name in ('original', 'control'))}
assert summary['eligible_for_replication'] == all(gates.values())
analysis = {'archive_files': archive_files, 'archive_sha256': digest(archive), 'source_and_checkpoint_hashes_pass': True,
    'skills_replayed_games': 384, 'skills_replayed_frames': replay_frames, 'arms': arm_tables, 'skills': skill_tables,
    'models': tables, 'comparisons': comparisons, 'gates': gates, 'decision': summary['decision'],
    'elapsed_seconds': summary['elapsed_seconds']}
dump(OUT / 'analysis_results.json', analysis)
print(json.dumps({**{k: v for k, v in analysis.items() if k not in ('arms', 'skills', 'models')},
    'arms': {a: {k: v for k, v in r.items() if k != 'validations'} for a, r in arm_tables.items()},
    'models': {n: {k: r[k] for k in ('active_win_rate', 'suicide_rate', 'draw_rate', 'counts')} for n, r in tables.items()}}, indent=2))
