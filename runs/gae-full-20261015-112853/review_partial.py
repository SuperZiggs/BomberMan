"""Audit a budget-limited B1 attempt without opening held-out games."""
from pathlib import Path
import gzip, json, math, sys, zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from audit_t2 import digest, dump
from audit_t3_approach import replay_labels
from benchmark_t3 import scenarios, summarize_games
OUT = Path(sys.argv[1]).resolve()
cfg = json.loads((OUT/'config.json').read_text())
s = json.loads((OUT/'summary.json').read_text())
assert s['status'] == 'incomplete' and s['decision'] == 'retain_original'
assert not s['eligible_for_replication'] and s['hashes_unchanged']
assert not (OUT/'test_opened.json').exists() and not list(OUT.glob('evaluation*.json'))
assert digest(Path(cfg['warm_start'])) == cfg['warm_start_sha256']
for name, expected in cfg['source_sha256'].items():
    assert digest(OUT/'source'/name) == expected
    if name not in ('run_gae_pilot.py','test_gae_pilot.py'):
        assert digest(ROOT/name) == expected
with zipfile.ZipFile(str(OUT)+'-send.zip') as z:
    assert z.testzip() is None
    for item in z.infolist():
        if not item.is_dir():
            assert z.read(item) == (OUT/item.filename).read_bytes()
    archive_files = sum(not item.is_dir() for item in z.infolist())
registry = {c['id']:c for c in scenarios(('validation',))}
reports, replay_frames, replay_games = {}, 0, 0
for arm, estimator in cfg['arms'].items():
    folder = OUT/arm
    rows = [json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
    vals = [json.loads(x) for x in (folder/'validation_metrics.jsonl').read_text().splitlines()]
    skills = [json.loads(x) for x in (folder/'skill_metrics.jsonl').read_text().splitlines()]
    games = [json.loads(x) for x in gzip.open(folder/'skill_games.jsonl.gz','rt')]
    assert len(games) == 64 * len(skills)
    for game in games:
        replay_frames += len(replay_labels(game,registry[game['scenario_id']]))
        replay_games += 1
    for probe in skills:
        tape = [g for g in games if g['episode'] == probe['episode']]
        assert probe['groups'] == {group:summarize_games([g for g in tape if g['group']==group]) for group in ('escape','approach')}
    assert len(rows) == s['arms'][arm]['completed_episodes']
    assert [r['episode'] for r in rows] == [r['optimizer_steps'] for r in rows] == list(range(1,len(rows)+1))
    assert s['arms'][arm]['optimizer_instances'] == 1
    for r in rows:
        d = r['advantage']
        assert not d['normalize_enabled'] and d['raw'] == d['policy']
        residual = d['mc_raw'] if estimator == 'gae' else d['raw']
        assert math.isclose(r['value_loss'], residual['mean']**2+residual['std']**2,rel_tol=1e-5,abs_tol=1e-5)
        assert math.isfinite(r['loss']) and math.isfinite(d['grad_norm_before_clip'])
        if estimator == 'gae':
            assert d['gae_lambda'] == .95 and d['terminal_transitions'] == 1
            assert d['bootstrap_transitions'] == d['continuations'] == d['samples']-1
    for cp in s['arms'][arm]['checkpoints'].values():
        assert digest(Path(cp['path'])) == cp['sha256']
        weights = torch.load(cp['path'],map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in weights.values())
    reports[arm] = {'episodes':len(rows),'learner_steps':sum(r['learner_steps'] for r in rows),
        'best_episode':s['arms'][arm]['best_episode'],'validations':vals,
        'skills':skills,'seconds':s['arms'][arm]['elapsed_seconds'],
        'clipped_updates':sum(r['advantage']['gradient_clipped'] for r in rows),
        'training_suicides':sum(r['suicide'] for r in rows),
        'advantage_std_mean':sum(r['advantage']['raw']['std'] for r in rows)/len(rows)}
analysis = {'status':'inconclusive','reason':s['reason'],'sources_and_B_unchanged_during_run':s['hashes_unchanged'],'archived_sources_verified':True,
    'post_run_timing_fix_only_files':['run_gae_pilot.py','test_gae_pilot.py'],
    'test_unopened':True,'archive_files':archive_files,'skills_replayed_games':replay_games,'skills_replayed_frames':replay_frames,'arms':reports,'elapsed_seconds':s['elapsed_seconds'],
    'decision':'retain_original'}
dump(OUT/'analysis_results.json',analysis)
print(json.dumps({**{k:v for k,v in analysis.items() if k!='arms'},'arms':{
    a:{k:v for k,v in r.items() if k not in ('validations','skills')} for a,r in reports.items()}},indent=2))
