from pathlib import Path
import json,sys,zipfile
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from audit_t2 import digest
from training_state import load_training_state
results={}
for label,name in (('cpu_resume','reward-resume-smoke-cpu-20261015'),('cuda','reward-resume-smoke-cuda-exec-20261015')):
    out=ROOT/'runs'/name
    cfg=json.loads((out/'config.json').read_text())
    s=json.loads((out/'summary.json').read_text())
    assert s['status']=='complete' and not s['full_protocol'] and s['hashes_unchanged']
    assert not s['eligible_for_replication'] and s['decision']=='retain_original'
    for source,sha in cfg['source_sha256'].items(): assert digest(out/'source'/source)==sha
    differing=[source for source,sha in cfg['source_sha256'].items() if digest(ROOT/source)!=sha]
    if label=='cuda': assert not differing
    else: assert set(differing)<={'training_state.py','run_reward_pilot.py','test_reward_pilot.py','test_training_state.py'}
    for arm in cfg['arms']:
        state=load_training_state(out/arm/'training_state.pt')
        assert state['episode']==state['scheduler']['last_epoch']==state['scheduler']['T_max']==20
        assert state['optimizer']['state'] and all(v['step'].item()==20 for v in state['optimizer']['state'].values())
        assert state['provenance']['runtime']==cfg['runtime']
        for source,sha in state['provenance']['sources'].items(): assert cfg['source_sha256'][source]==sha
        rows=[json.loads(x) for x in (out/arm/'metrics.jsonl').read_text().splitlines()]
        assert [r['optimizer_steps'] for r in rows]==list(range(1,21))
        assert state['context']['rows']==rows
        assert state['histories']['reward']==[r['reward'] for r in rows]
        final=torch.load(out/arm/'final_phaseC.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(v,final[k]) for k,v in state['model'].items())
        assert digest(out/arm/'training_state.pt')==s['arms'][arm]['training_state']['sha256']
    with zipfile.ZipFile(str(out)+'-send.zip') as archive:
        assert archive.testzip() is None
        for item in archive.infolist():
            if not item.is_dir(): assert archive.read(item)==(out/item.filename).read_bytes()
    results[label]={'status':'reviewed','episodes_per_arm':20,'full_protocol':False,'source_changes_after_completion':differing}
proof=json.loads((ROOT/'runs/resume-implementation-20261015/control_before_resume.json').read_text())
out=ROOT/'runs/reward-resume-smoke-cpu-20261015'
assert all(digest(out/name)==sha for name,sha in proof.items())
results['cpu_resume']['completed_control_files_byte_identical']=len(proof)
(ROOT/'runs/resume-implementation-20261015/smoke_review.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
