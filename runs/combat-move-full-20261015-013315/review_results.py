"""Recompute the completed pilot's provenance, metrics, skills and paired gates."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
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
assert config['train_seed'] == 2026101501 and config['episodes_per_arm'] == 200
assert config['evaluation']['seed'] == 33000 and config['evaluation']['games_per_model'] == 400
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
arm_tables, skill_tables, replay_frames = {}, {}, 0
for arm, teacher in config['arms'].items():
    folder = OUT / arm
    cfg = json.loads((folder / 'config.json').read_text())
    assert cfg['attack_teacher'] == teacher and cfg['resume_from'] == config['warm_start']
    assert (cfg['episodes'], cfg['seed'], cfg['lr'], cfg['max_steps']) == (200, 2026101501, 1e-4, 500)
    rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    assert [r['episode'] for r in rows] == list(range(1, 201))
    assert [r['optimizer_steps'] for r in rows] == list(range(1, 201))
    assert all(r['learning_aid'] == (r['episode'] - 1) % 2 for r in rows)
    guidance, rejections = Counter(), Counter()
    for r in rows:
        g = r['move_guidance']
        assert r['attack_teacher'] == teacher and 0 < r['learner_steps'] <= 500
        assert 0 <= g['agreements'] <= g['labels'] and 0 <= g['accepted_agreements'] <= g['accepted'] <= g['queries']
        assert g['queries'] == (g['labels'] if arm == 'combat_move' else 0)
        assert sum(g['rejected'].values()) == g['queries'] - g['accepted']
        assert 0 <= g['target_probability_sum'] <= g['labels'] + 1e-5
        assert 0 <= g['entropy_sum'] <= g['labels'] * np.log(6) + 1e-4
        assert 0 <= g['pressure_gain_sum'] <= g['accepted'] + 1e-5
        assert r['combat_bombs'] <= r['combat_labels'] - g['accepted']
        guidance.update({k: v for k, v in g.items() if k != 'rejected'})
        rejections.update(g['rejected'])
    assert {**guidance, 'rejected': dict(rejections)} == summary['arms'][arm]['guidance']
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
        frames = replay_labels(game, registry[game['scenario_id']])
        replay_frames += len(frames)
    per_probe = {}
    for probe in probes:
        selected_games = [g for g in games if g['episode'] == probe['episode']]
        assert len(selected_games) == 64
        recomputed = {group: summarize_games([g for g in selected_games if g['group'] == group]) for group in ('escape', 'approach')}
        assert recomputed == probe['groups']
        per_probe[str(probe['episode'])] = recomputed
    for checkpoint, episode in (('best', best), ('final', 200)):
        net = ActorCriticNet(board_size=9)
        path = folder / f'{checkpoint}_phaseC.pt'
        smart_load_weights(net, str(path), 'cpu')
        assert model_digest(net) == next(p['model_state_sha256'] for p in probes if p['episode'] == episode)
        assert digest(path) == summary['arms'][arm]['checkpoints'][checkpoint]['sha256']
    skill_tables[arm] = per_probe
    labels, accepted = guidance['labels'], guidance['accepted']
    arm_tables[arm] = {'best_episode': best, 'learner_steps': sum(r['learner_steps'] for r in rows),
        'training_suicides': sum(r['suicide'] for r in rows), 'training_active_wins': sum(r['active_win'] for r in rows),
        'guidance': {**guidance, 'rejected': dict(rejections)},
        'move_agreement_percent': 100 * guidance['agreements'] / labels,
        'bomb_instead_of_move_percent': 100 * guidance['bomb_instead_of_move'] / labels,
        'mean_target_probability': guidance['target_probability_sum'] / labels,
        'mean_entropy': guidance['entropy_sum'] / labels,
        'coverage_percent': 100 * accepted / max(1, guidance['queries']),
        'accepted_agreement_percent': 100 * guidance['accepted_agreements'] / accepted if accepted else None,
        'mean_pressure_gain': guidance['pressure_gain_sum'] / accepted if accepted else None,
        'mean_query_ms': guidance['milliseconds'] / guidance['queries'] if guidance['queries'] else None,
        'validations': vals, 'seconds': summary['arms'][arm]['elapsed_seconds']}

models, tables = {}, {}
for name in ('original', 'control', 'combat_move'):
    r = json.loads((OUT / f'evaluation_{name}_33000.json').read_text())
    assert (r['seed'], r['num_episodes'], r['num_agents'], r['max_steps']) == (33000, 400, 2, 500)
    assert [(g['seed'], g['aid']) for g in r['games']] == [(s, a) for s in range(33000, 33200) for a in (0, 1)]
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
comparisons = {name: paired_active_bootstrap(models['combat_move'], models[name]) for name in ('original', 'control')}
assert comparisons == summary['comparisons']
gates = {'gain': all(c['difference_pp'] >= 5 for c in comparisons.values()),
         'ci': all(c['ci95_pp'][0] > 0 for c in comparisons.values()),
         'suicide': all(models['combat_move']['suicide_rate'] <= models[name]['suicide_rate'] for name in ('original', 'control'))}
assert summary['eligible_for_replication'] == all(gates.values())
analysis = {'archive_files': archive_files, 'archive_sha256': digest(archive), 'source_and_checkpoint_hashes_pass': True,
            'skills_replayed_games': 384, 'skills_replayed_frames': replay_frames,
            'arms': arm_tables, 'skills': skill_tables, 'models': tables, 'comparisons': comparisons, 'gates': gates,
            'decision': summary['decision'], 'elapsed_seconds': summary['elapsed_seconds']}
dump(OUT / 'analysis_results.json', analysis)
print(json.dumps({'arms': {a: {k: v for k, v in r.items() if k not in ('validations', 'guidance')} for a, r in arm_tables.items()},
                  'models': {n: {k: r[k] for k in ('active_win_rate', 'suicide_rate', 'draw_rate', 'counts')} for n, r in tables.items()},
                  'comparisons': comparisons, 'gates': gates, 'decision': analysis['decision'],
                  'skills_replayed_frames': replay_frames}, indent=2))
