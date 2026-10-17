"""Recover completed R1 records after process termination; never resume training or open test."""
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from audit_t2 import digest, dump
from audit_t3_approach import replay_labels
from benchmark_t3 import scenarios, summarize_games
from run_t3_curriculum import model_digest
from train_actor_critic import ActorCriticNet, smart_load_weights

torch.set_num_threads(1)
OUT = Path(sys.argv[1]).resolve()
assert not (OUT / 'summary.json').exists()
assert not (OUT / 'test_opened.json').exists() and not list(OUT.glob('evaluation*.json'))
cfg = json.loads((OUT / 'config.json').read_text(encoding='utf-8-sig'))
assert digest(Path(cfg['warm_start'])) == cfg['warm_start_sha256']
for name, expected in cfg['source_sha256'].items():
    assert digest(ROOT / name) == digest(OUT / 'source' / name) == expected
for path, expected in cfg['preflight_input_sha256'].items():
    assert digest(Path(path)) == expected

def records(path):
    text = path.read_text(encoding='utf-8-sig')
    lines = text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
    return [json.loads(line) for line in lines]

registry = {case['id']: case for case in scenarios(('validation',))}
arms, details, frames, games_checked = {}, {}, 0, 0
for arm, coefficient in cfg['arms'].items():
    folder = OUT / arm
    rows = records(folder / 'metrics.jsonl')
    vals = records(folder / 'validation_metrics.jsonl')
    skills = records(folder / 'skill_metrics.jsonl')
    assert [row['episode'] for row in rows] == [row['optimizer_steps'] for row in rows] == list(range(1, len(rows) + 1))
    for row in rows:
        assert math.isclose(sum(row['reward_components'].values()), row['reward'], rel_tol=0, abs_tol=1e-10)
        assert row['reward_component_max_error'] <= 1e-10
        count = row['reward_components'].get('wall', 0) / coefficient
        assert math.isclose(count, round(count), rel_tol=0, abs_tol=1e-9)
        assert 0 <= round(count) <= row['walls']
        d = row['advantage']
        assert not d['normalize_enabled'] and d['raw'] == d['policy']
        assert math.isclose(row['value_loss'], d['raw']['mean']**2 + d['raw']['std']**2, rel_tol=1e-5, abs_tol=1e-5)
        assert all(math.isfinite(row[key]) for key in ('loss', 'policy_loss', 'value_loss', 'reward'))
        assert row['contact']['min_distance'] is not None
    decoder = zlib.decompressobj(31)
    text = decoder.decompress((folder / 'skill_games.jsonl.gz').read_bytes()).decode('utf-8')
    lines = text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
    games = [json.loads(line) for line in lines]
    assert len(games) == 64 * len(skills)
    for game in games:
        frames += len(replay_labels(game, registry[game['scenario_id']]))
        games_checked += 1
    for probe in skills:
        selected = [game for game in games if game['episode'] == probe['episode']]
        assert probe['groups'] == {group: summarize_games([g for g in selected if g['group'] == group])
                                   for group in ('escape', 'approach')}
    checkpoints = {}
    best = json.loads((folder / 'best_validation.json').read_text())['episode']
    for kind in ('best', 'final'):
        path = folder / f'{kind}_phaseC.pt'
        if path.exists():
            net = ActorCriticNet(board_size=9)
            smart_load_weights(net, str(path), 'cpu')
            assert all(torch.isfinite(value).all() for value in net.state_dict().values())
            episode = best if kind == 'best' else 200
            assert model_digest(net) == next(p['model_state_sha256'] for p in skills if p['episode'] == episode)
            checkpoints[kind] = {'path': str(path), 'sha256': digest(path)}
    complete = len(rows) == 200
    arms[arm] = {'completed_episodes': len(rows), 'expected_episodes': 200,
        'optimizer_steps_logged': rows[-1]['optimizer_steps'], 'training_complete': complete,
        'validation_complete': [v['episode'] for v in vals] == [0, 100, 200],
        'skill_probes_complete': [v['episode'] for v in skills] == [0, 100, 200],
        'best_episode': best, 'checkpoints': checkpoints, 'last_logged_elapsed_seconds': rows[-1]['elapsed_seconds']}
    details[arm] = {'learner_steps_logged': sum(r['learner_steps'] for r in rows), 'validations': vals,
        'skills': skills, 'skill_gzip_complete': decoder.eof, 'recovered_skill_games': len(games),
        'reward_component_error_max': max(r['reward_component_max_error'] for r in rows)}
assert arms['control']['completed_episodes'] == 200 and arms['half_wall']['completed_episodes'] == 199
assert arms['half_wall']['best_episode'] == 100 and 'final' not in arms['half_wall']['checkpoints']
reason = 'process_terminated_during_half_wall_validation_200'
summary = {'status': 'incomplete', 'full_protocol': True, 'reason': reason, 'arms': arms,
    'evaluation_complete': False, 'hashes_unchanged': True, 'comparisons': {}, 'models': {},
    'eligible_for_replication': False, 'decision': 'retain_original', 'elapsed_seconds': None,
    'last_console_output_utc': datetime.fromtimestamp(Path(str(OUT) + '-console.log').stat().st_mtime, timezone.utc).isoformat(),
    'finalized_after_interruption_utc': datetime.now(timezone.utc).isoformat(),
    'episode_200_half_wall_update_not_recoverable': True, 'archive_created_by': 'post_interruption_audit'}
analysis = {'status': 'reviewed_incomplete', 'decision': 'retain_original', 'reason': reason,
    'source_checkpoint_and_r0_input_hashes_pass': True, 'heldout_test_opened': False,
    'skills_replayed_games': games_checked, 'skills_replayed_frames': frames, 'arms': details,
    'limit': 'Only completed JSON records recovered; truncated skill gzip footer is reported, original bytes unchanged.'}
dump(OUT / 'summary.json', summary)
dump(OUT / 'analysis_results.json', analysis)
print(json.dumps({'summary': summary, 'replay_games': games_checked, 'replay_frames': frames,
    'skill_gzip_complete': {arm: data['skill_gzip_complete'] for arm, data in details.items()}}, indent=2))
