"""Independently verify D0 artifacts without training or evaluating another model."""
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from audit_t2 import digest, dump
from training_state import load_training_state

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
paths = read(HERE/'smoke-paths.json')
review = {}
for device, relative in paths.items():
    out = ROOT/relative
    cfg, summary, parity = (read(out/name) for name in ('config.json', 'summary.json', 'first_update_parity.json'))
    assert summary['status'] == 'complete' and not summary['full_protocol'] and summary['hashes_unchanged']
    assert cfg['evaluation'] is None and cfg['device'] == device and cfg['estimator'] == 'mc'
    assert digest(Path(cfg['warm_start'])) == cfg['warm_start_sha256']
    for name, expected in cfg['source_sha256'].items():
        assert digest(ROOT/name) == digest(out/'source'/name) == expected
    assert not list(out.glob('evaluation*.json')) and not (out/'test_opened.json').exists()
    with zipfile.ZipFile(Path(str(out)+'-send.zip')) as z:
        assert z.testzip() is None
        for item in z.infolist():
            if not item.is_dir():
                assert z.read(item) == (out/item.filename).read_bytes(), item.filename
        files = sum(not item.is_dir() for item in z.infolist())
    modes = {}
    for mode in ('ac', 'ratio'):
        folder = out/mode
        arm_cfg = read(folder/'config.json')
        assert arm_cfg['policy_objective'] == mode and arm_cfg['advantage_estimator'] == 'mc'
        assert not arm_cfg['normalize_advantage']
        assert arm_cfg['validation_every'] == 0 and not arm_cfg['run_report']
        rows = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [r['episode'] for r in rows] == [r['optimizer_steps'] for r in rows] == list(range(1,21))
        assert all(r['learning_aid'] == (r['episode']-1)%2 and r['reward_component_max_error'] <= 1e-10 for r in rows)
        state = load_training_state(folder/'training_state.pt')
        assert digest(folder/'training_state.pt') == summary['modes'][mode]['training_state_sha256']
        assert state['episode'] == state['scheduler']['last_epoch'] == state['scheduler']['T_max'] == 20
        assert state['provenance']['config']['policy_objective'] == mode
        assert state['provenance']['config']['wall_destroy_bonus'] == .15
        assert state['provenance']['runtime'] == cfg['runtime']
        assert set(state['rng']) == {'python','numpy','env','torch','cuda'}
        assert {int(s['step'].item()) for s in state['optimizer']['state'].values()} == {20}
        assert state['histories']['reward'] == [r['reward'] for r in rows]
        for name, expected in state['provenance']['sources'].items():
            assert cfg['source_sha256'][name] == expected
        weights = torch.load(folder/'final_phaseC.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(weights[k], v) and torch.isfinite(v).all().item() for k,v in state['model'].items())
        for name, expected in summary['modes'][mode]['checkpoints'].items():
            assert digest(folder/f'{name}_phaseC.pt') == expected
        if mode == 'ratio':
            assert all(r['ratio_update']['samples'] == r['learner_steps'] and r['ratio_update']['epochs'] == 1
                and not r['ratio_update']['clipped'] and r['ratio_update']['terminal_transitions'] == 1 for r in rows)
        modes[mode] = {'episodes':len(rows), 'adam_steps':20, 'learner_steps':sum(r['learner_steps'] for r in rows),
                       'rng_scheduler_history_and_weights_verified':True}
    assert parity['actions_and_rewards_identical']
    assert parity['ratio']['ratio_max_abs_error_from_one'] < 1e-4
    assert max(parity['auxiliary_and_value_loss_abs_delta'].values()) <= 2e-5
    review[device] = {'status':'reviewed', 'path':str(out), 'raw_archive_files':files, 'modes':modes,
        'first_update':parity, 'seconds':summary['elapsed_seconds'], 'peak_cuda_bytes':{
            mode:data['peak_cuda_bytes'] for mode,data in summary['modes'].items()}}
dump(HERE/'analysis_results.json', review)
print(json.dumps(review,indent=2))
