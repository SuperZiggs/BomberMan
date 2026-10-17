"""Package the complete reviewed D1 trial, keeping the runner's raw ZIP untouched."""
import difflib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from audit_t2 import digest,dump

out=Path(sys.argv[1]).resolve()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
cfg,s,a=(read(out/name) for name in ('config.json','summary.json','analysis_results.json'))
assert s['status']=='complete' and s['full_protocol'] and s['hashes_unchanged']
assert a['source_and_checkpoint_hashes_pass'] and len(a['training_state_checks'])==2
for name,expected in cfg['source_sha256'].items():assert digest(ROOT/name)==digest(out/'source'/name)==expected
assert digest(Path(cfg['warm_start']))==cfg['warm_start_sha256']
for arm,info in s['arms'].items():
    for item in (*info['checkpoints'].values(),info['training_state']):assert digest(Path(item['path']))==item['sha256']
evidence=out/'verification'
evidence.mkdir(exist_ok=False)
for name in ('all-tests.log','metadata-compat-tests.log','d1-tests.log','runner-tests.log',
             'smoke_review.json','smoke-review.log','full-review.log','profile500-console.log',
             'before-resume.json','resume_review.json'):
    shutil.copy2(HERE/name,evidence/name)
for name in ('review_results.py','write_report.py'):
    shutil.copy2(HERE/name,out/name)
shutil.copy2(HERE/'check_resume.py',evidence/'check_resume.py')
for name in ('profile_500.py','profile500/summary.json','profile500/ratio/metrics.json','profile500/ppo/metrics.json'):
    target=evidence/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(HERE/name,target)
(evidence/'docs').mkdir()
for name in ('PPO_PILOT_GUIDE.md','PLAN_BOMBERMAN.md','README.md','IMPROVEMENTS.md','RESULTS.md'):
    shutil.copy2(ROOT/name,evidence/'docs'/name)
for name in ('train_actor_critic.py','run_combat_move_pilot.py','test_ppo_d0.py'):
    before=(HERE/'before'/name).read_text(encoding='utf-8-sig').splitlines(True)
    after=(ROOT/name).read_text(encoding='utf-8-sig').splitlines(True)
    (evidence/(name+'.diff')).write_text(''.join(difflib.unified_diff(before,after,fromfile='before/'+name,tofile=name)),encoding='utf-8')
shutil.copy2(Path(str(out)+'-console.log'),out/'console.log')
shutil.copy2(Path(str(out)+'-resume-console.log'),out/'resume-console.log')
(evidence/'EVIDENCE.md').write_text('''# D1 evidence

The raw `-send.zip` is unchanged. This review archive adds independent results, report, console, tests, smoke checks and document snapshots.

241 regression tests passed. A final metadata-only fix made per-epoch objective/clipped flags agree with PPO; 25 focused metadata/compatibility tests then passed. Counts overlap and are not 266 distinct tests. Model/update arithmetic was unchanged by that metadata fix.

D1 tests include sign-sensitive clipped surrogate, fixed targets and group means, common post-update KL stopping, nonfinite gradients, cancellation restoring previous Adam/model, and exact PPO resume in new Python processes on CPU/CUDA. The full trial started fresh, then P4 resumed from episode36/Adam144 after the user freed VRAM. U4 completed files stayed byte-identical. The original deadlines were retained; both console logs and resume audit are included.

The 500-step fixture forces STOP and disables KL stopping to exercise all four updates; it is a memory/finite-value check, not a full-game throughput or policy evaluation. The first mode also includes cold runtime initialization costs and may overlap a finishing compatibility test.

All full-state checkpoints are for the registered Phase C config/source/runtime/device. Resume preserves original deadlines. Best/final weights are for inference; full state is for training recovery. A completed pilot needs no resume.

Only one train seed was used. CI resamples paired board seeds with both corners, not independent games or independent training runs. No automatic model replacement occurred.
''',encoding='utf-8')
archive=Path(str(out)+'-review.zip')
assert not archive.exists()
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(out).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for item in z.infolist():assert z.read(item)==(out/item.filename).read_bytes()
    files=len(z.infolist())
dump(HERE/'package_review.json',{'status':'verified','files':files,'path':str(archive),
    'sha256':digest(archive),'raw_archive_sha256':digest(Path(str(out)+'-send.zip')),
    'baseline_and_sources_unchanged':True})
print(archive)
