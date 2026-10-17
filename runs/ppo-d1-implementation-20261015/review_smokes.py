"""Check completed CPU/CUDA D1 smoke states, validation, raw ZIPs and locked models."""
import gzip
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
from audit_t2 import digest,dump
from run_ppo_pilot import verify_updates
from training_state import load_training_state

read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
paths=read(HERE/'smoke-paths.json')
review={}
for device,relative in paths.items():
    out=ROOT/relative
    cfg,s=read(out/'config.json'),read(out/'summary.json')
    assert s['status']=='complete' and not s['full_protocol'] and s['hashes_unchanged']
    assert not s['eligible_for_replication'] and s['decision']=='retain_original'
    assert cfg['evaluation']['seed']==39870 and cfg['train_device']==device
    assert digest(Path(cfg['warm_start']))==cfg['warm_start_sha256']
    for name,expected in cfg['source_sha256'].items():
        assert digest(ROOT/name)==digest(out/'source'/name)==expected
    verify_updates(out,s['arms'])
    with zipfile.ZipFile(Path(str(out)+'-send.zip')) as z:
        assert z.testzip() is None
        for item in z.infolist():
            if not item.is_dir():assert z.read(item)==(out/item.filename).read_bytes()
        files=sum(not item.is_dir() for item in z.infolist())
    modes={}
    for arm in ('control','ppo'):
        folder=out/arm
        rows=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
        vals=[json.loads(x) for x in (folder/'validation_metrics.jsonl').read_text().splitlines()]
        probes=[json.loads(x) for x in (folder/'skill_metrics.jsonl').read_text().splitlines()]
        games=[json.loads(x) for x in gzip.open(folder/'skill_games.jsonl.gz','rt')]
        state=load_training_state(folder/'training_state.pt')
        assert state['episode']==state['scheduler']['last_epoch']==20
        assert [v['episode'] for v in vals]==[p['episode'] for p in probes]==[0,10,20]
        assert len(rows)==20 and len(games)==192
        assert state['context']['rows']==rows and state['context']['validation_rows']==vals
        assert state['context']['probes']==probes and state['context']['games']==games
        assert state['provenance']['runtime']==cfg['runtime']
        assert set(state['rng'])=={'python','numpy','env','torch','cuda'}
        assert digest(folder/'training_state.pt')==s['arms'][arm]['training_state']['sha256']
        for kind,expected in (('final',state['model']),('best',state['selection'][4])):
            weights=torch.load(folder/f'{kind}_phaseC.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(v,weights[k]) for k,v in expected.items())
        modes[arm]={'episodes':20,'adam_steps':s['arms'][arm]['optimizer_steps'],
            'early_stop_episodes':sum(r['ratio_update']['early_stop'] for r in rows),
            'source_state_context_rng_weights_pass':True}
    selected=read(out/'selected_models.json')
    assert all(digest(Path(r['path']))==r['sha256'] for r in selected.values())
    review[device]={'status':'reviewed','raw_files':files,'modes':modes,'seconds':s['elapsed_seconds']}
dump(HERE/'smoke_review.json',review)
print(json.dumps(review,indent=2))
