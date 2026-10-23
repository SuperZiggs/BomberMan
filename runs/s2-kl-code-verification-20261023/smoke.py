"""Bounded, smoke-only CLI evidence driver. Never registers or runs the full pilot."""
import argparse, hashlib, json, subprocess, sys, time, zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--device',choices=('cpu','cuda:0'),required=True)
parser.add_argument('--suffix',default='')
args=parser.parse_args()
label='cpu' if args.device=='cpu' else 'cuda'
output=root/'runs'/('s2-kl-code-smoke-20261023-'+label+args.suffix)
evidence=Path(__file__).resolve().parent
assert not output.exists(), 'Keep existing smoke output immutable; use another suffix'
protected=json.loads((evidence/'protected-before.json').read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert all(digest(root/p)==sha for p,sha in protected.items())
stages=[('register','--input-root',str(root),'--device',args.device,'--workers','2','--smoke'),
        ('train','--arm','control'),('train','--arm','anchor'),('diagnose',),('validate',),('package',)]
measurements=[]
for index,stage in enumerate(stages):
    name=stage[0]+('-'+stage[2] if stage[0]=='train' else '')
    command=[str(root/'venv/Scripts/python.exe'),'-u',str(root/'run_s2_kl_probe.py'),*stage,'--output',str(output)]
    log=evidence/('smoke-'+label+args.suffix+'-'+name+'.log')
    started=time.perf_counter()
    with log.open('w',encoding='utf-8') as handle:
        result=subprocess.run(command,cwd=root,stdout=handle,stderr=subprocess.STDOUT,check=False)
    elapsed=time.perf_counter()-started
    if result.returncode:
        print(log.read_text(encoding='utf-8',errors='replace')[-6000:],flush=True)
        raise SystemExit('Smoke stage failed: '+name+' exit '+str(result.returncode))
    measured=json.loads((output/'logs/stages.jsonl').read_text(encoding='utf-8').splitlines()[-1])
    measurements.append(dict(stage=name,cli_wall_seconds=elapsed,process=measured))
    print(label+' '+name+' PASS '+f'{elapsed:.2f}s',flush=True)
report=json.loads((output/'report.json').read_text(encoding='utf-8'))
manifest=json.loads((output/'model_manifest.json').read_text(encoding='utf-8'))
assert report['complete'] and report['status']=='SMOKE_ONLY'
assert report['counts']['games']==report['counts']['expected_games']==4*len(manifest['models'])
workers={}
for path in (output/'validation/tasks').glob('*.json'):
    board=json.loads(path.read_text(encoding='utf-8'))
    assert len(board['records'])==2 and sorted(r['corner'] for r in board['records'])==[0,1]
    worker=board['worker_runtime']
    assert worker['device']=='cpu' and worker['threads']==1 and worker['deterministic']
    assert Path(worker['source_root']).resolve()==(output/'sources').resolve()
    assert all(r['board'] in (790002,790003) for r in board['records'])
    if worker.get('peak_rss_bytes') is not None:
        workers[str(worker['pid'])]=max(workers.get(str(worker['pid']),0),worker['peak_rss_bytes'])
assert not (output/'test_opened.json').exists()
with zipfile.ZipFile(str(output)+'-send.zip') as archive:
    assert archive.testzip() is None
assert all(digest(root/p)==sha for p,sha in protected.items())
assert not (root/'runs/s2-kl-20261023').exists()
record=dict(output=str(output),device=args.device,status=report['status'],counts=report['counts'],
    measurements=measurements,worker_peak_rss_bytes=workers,protected_unchanged=True,full_run_started=False)
(evidence/('smoke-'+label+args.suffix+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
print('SMOKE VERIFIED '+str(output),flush=True)
