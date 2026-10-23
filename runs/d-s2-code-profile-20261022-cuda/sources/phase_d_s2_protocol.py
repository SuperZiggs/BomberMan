"""Immutable inputs and registration for the controlled D-S2 experiment."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent
SPEC = 'docs/superpowers/specs/2026-10-22-phase-d-s2-pool-design.md'
CHECKPOINTS = {
    'S2': ('runs/sd8a-20261020/train/epoch_16.pt', 'd222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814'),
    'G': ('runs/sd75-20261018/full/best_phaseC.pt', 'bb0c775c58161aa1bbdd229b9528170aab174d5aabf8f4a6739b848919cdc7ac'),
    'F': ('runs/phaseC-sd-best-fallback/best_phaseC.pt', '0315a56f09079333d936db5f55587093f1b6415ed6355d2ef6ae2ab63f8b1483'),
    'S': ('runs/sd53-20261018/train/epoch_8.pt', '8f7116d290593fa10de1753c4f0f96366e220a193e9860123ed72e0db46bfd4a'),
    'B': ('runs/pressure-C-next/final_phaseC.pt', '93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60'),
}
CNN = ('G', 'F', 'S', 'S2')
_DEVICE = 'cpu'


def config_for(smoke=False):
    train = dict(phase='D', episodes=2000, board_size=9, num_agents=2, max_steps=500,
        gamma=.99, lr=.000025, value_coef=.5, entropy_coef=.0075, wall_destroy_bonus=.15,
        snapshot_every=200, seed=2026102201, validation_every=0, validation_episodes=2,
        attack_teacher='local', teacher_version='v2', normalize_advantage=False,
        advantage_estimator='mc', policy_objective='ac', sudden_death_probability=1.,
        sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40,
        attack_lookahead=2, min_pressure_gain=.15, trap_sequence_depth=2,
        sequence_pressure_gain=.35, escape_loss_coef=.5, combat_loss_coef=.9,
        navigation_loss_coef=.4, ring_safety_coef=.6, late_game_start=350,
        late_game_attack_multiplier=1.75, search_teacher=False, group_rl=False,
        anchor_coef=0., run_report=False, log_every=50)
    if smoke:
        train.update(episodes=4, max_steps=20, snapshot_every=2, log_every=4)
    return dict(train=train, pool_slots=['RB', 'G', 'F', 'S', 'S2', 'RB'] + ['recent']*4,
        candidate_episodes=[0, 2, 4] if smoke else [0, 500, 1000, 1500, 2000],
        validation_games=2 if smoke else 80, test_games=2 if smoke else 200, workers=2)


canonical_config = config_for


def canonical_digest(value, digest_key=None):
    if digest_key is not None:
        value = {k: v for k, v in value.items() if k != digest_key}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def sha(path):
    h = hashlib.sha256()
    try:
        with Path(path).open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
    except OSError as e:raise ValueError('Missing/unreadable registered artifact: '+str(path)) from e
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
        parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+v)))


def atomic_json(path, payload, *, overwrite=False):
    path = Path(path)
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(str(path))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def configure_runtime(device='cpu'):
    global _DEVICE
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    if device not in ('cpu', 'cuda', 'cuda:0'):
        raise ValueError('Unsupported device')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    _DEVICE=device


def runtime():
    from training_state import runtime_state
    return dict(python=sys.version, platform=platform.platform(), device=_DEVICE, **runtime_state())


def seed_manifest(smoke):
    ranges = ((690000,690001),(690002,690002),(690003,690003),(690003,690003)) if smoke else (
        (610000,610999),(620000,620039),(630000,630099),(640000,640099))
    return {'mode': 'smoke' if smoke else 'full', 'ranges': {k: dict(start=a,end_inclusive=b)
        for k,(a,b) in zip(('train','validation','test_reference','test_snapshots'),ranges)}}


def source_closure():
    pending = ['run_phase_d_s2', 'train_actor_critic', 'training_state', 'evaluate',
               'bomberman_env','rule_based_bot','tactical_teacher','combat_move_teacher',
               'kill_search','group_rl','progress']
    found = {}
    while pending:
        name = pending.pop()
        path = ROOT/(name+'.py')
        if name+'.py' in found or not path.is_file():
            continue
        found[name+'.py'] = sha(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for imported in names:
                local = imported.split('.')[0]
                if (ROOT/(local+'.py')).is_file():
                    pending.append(local)
    return dict(sorted(found.items()))


def inventory_seeds(input_root,exclude=()):
    """Read seed-bearing metadata; unknown registration formats fail closed."""
    root = Path(input_root)/'runs'
    ranges, unresolved, sources, rng_seeds = [], [], {}, []
    def visit(v, source):
        if isinstance(v,dict):
            if 'start' in v and ('end_inclusive' in v or 'end' in v):
                a,b = v['start'],v.get('end_inclusive',v.get('end'))
                if isinstance(a,int) and isinstance(b,int): ranges.append([a,b,source])
            for k,x in v.items():
                if k=='seed' and isinstance(x,int) and 'phase' in v and 'episodes' in v:
                    rng_seeds.append(dict(source=source,seed=x,kind='training_rng'))
                elif k in ('board','board_seed','seed','normal_seed','validation_seed','test_seed','train_seed','mechanism_seed','collect_seed') and isinstance(x,int):
                    ranges.append([x,x,source])
                    if k in ('validation_seed','test_seed','collect_seed','mechanism_seed'):
                        count=v.get(k.replace('_seed','_games'),v.get(k.replace('_seed','_episodes')))
                        if isinstance(count,int) and count>0:ranges.append([x,x+count-1,source])
                elif k.endswith('_board_seeds') and isinstance(x,list) and x:
                    if all(isinstance(n,int) for n in x): ranges.append([min(x),max(x),source])
                visit(x,source)
        elif isinstance(v,list):
            for x in v:visit(x,source)
    names = {'registration.json','config.json','test_opened.json','seed_manifest.json',
             'pde_seed_registry.json','report.json'}
    if root.exists():
        for path in sorted([*root.rglob('*.json'),*root.rglob('*.jsonl')]):
            if any(path.resolve().is_relative_to(Path(p).resolve()) for p in exclude):continue
            if path.name=='phase_d_s2_seed_registry.json':continue
            if path.name not in names and path.suffix!='.jsonl' and not any(s in path.name.lower() for s in ('raw','board','record','evaluation')):continue
            rel = str(path.relative_to(root))
            if path.suffix=='.jsonl':
                try:
                    with path.open(encoding='utf-8') as handle:
                        for line in handle:
                            if line.strip():visit(json.loads(line),rel)
                    sources[rel]=sha(path)
                except (ValueError,OSError) as e:unresolved.append(rel+': '+str(e))
                continue
            # Large raw artifacts require a separately verified seed index, not unbounded RAM.
            if path.stat().st_size>32*1024*1024:
                unresolved.append(rel+': requires seed index (metadata >32MB)');continue
            try:data=read_json(path)
            except (ValueError,OSError) as e:unresolved.append(rel+': '+str(e));continue
            sources[rel]=sha(path)
            before=len(ranges);visit(data,rel)
            if path.name=='registration.json' and len(ranges)==before and 'config' not in data:
                unresolved.append(rel+': unknown seed mapping')
            if path.name=='registration.json' and data.get('mechanism_seed')==6700:
                # SD7.4's frozen runner explicitly registers 100 corner-paired games.
                snapshot=path.parent/'source/run_sd74_probe.py'
                expected=data.get('sources',{}).get('run_sd74_probe.py')
                # Old Git checkouts converted LF to CRLF. Record raw bytes in inventory;
                # accept normalized history only, never normalize current-run source hashes.
                historical=hashlib.sha256(snapshot.read_bytes().replace(b'\r\n',b'\n')).hexdigest() if snapshot.is_file() else None
                if expected and snapshot.is_file() and expected in (sha(snapshot),historical):
                    tree=ast.parse(snapshot.read_text(encoding='utf-8-sig'))
                    verified=any(isinstance(n,ast.Assign) and isinstance(n.value,ast.Tuple) and
                        [e.value for e in n.value.elts if isinstance(e,ast.Constant)][:2]==[6700,100] and
                        any(isinstance(t,ast.Tuple) and [e.id for e in t.elts if isinstance(e,ast.Name)][:2]==['MECH_SEED','MECH_GAMES'] for t in n.targets)
                        for n in ast.walk(tree))
                    if verified:ranges.append([6700,6749,rel])
                    else:unresolved.append(rel+': unverified mechanism range')
                else:unresolved.append(rel+': missing frozen SD7.4 runner')
            if path.name=='registration.json' and data.get('normal_seed') in (50000,51000):
                base=data['normal_seed']
                spans=((0,39),(1000,1059),(2000,2099)) if base==50000 else (
                    (0,49),(1000,1049),(2000,2399),(3000,3399),(4000,4099),(5000,5099))
                ranges.extend([base+a,base+b,rel] for a,b in spans)
    return dict(ranges=ranges,rng_seeds=rng_seeds,unresolved=unresolved,sources=sources)


def _check_inputs(input_root):
    result={}
    for name,(relative,expected) in CHECKPOINTS.items():
        path=Path(input_root)/relative
        if not path.is_file() or sha(path)!=expected:raise ValueError('Checkpoint hash mismatch: '+name)
        result[name]=dict(id=name,checkpoint=str(path.resolve()),sha256=expected,episode=None,role='reference')
    selected=read_json(Path(input_root)/'runs/sd8a-20261020/selected.json')
    opened=read_json(Path(input_root)/'runs/sd8a-20261020/test_opened.json')
    if Path(str(selected['checkpoint']).replace('\\','/')).as_posix()!=CHECKPOINTS['S2'][0] or opened['models']['S2']!=CHECKPOINTS['S2'][1]:
        raise ValueError('S2 selection/test-open hash mismatch')
    result['RB']=dict(id='RB',checkpoint=None,sha256=sha(ROOT/'rule_based_bot.py'),episode=None,role='anchor')
    return result


def register(output,input_root,device='cpu',workers=2,smoke=False,resume=False):
    from training_state import run_lock
    output,input_root=Path(output).resolve(),Path(input_root).resolve()
    if workers not in range(1,6):raise ValueError('workers must be 1..5')
    configure_runtime(device)
    with run_lock(output):
        if output.exists():
            if not resume:raise FileExistsError('Output already exists')
            reg=verify_run(output,allow_pending=True)
            if (reg['input_root']!=str(input_root) or reg['device']!=device or reg['mode']!=('smoke' if smoke else 'full')
                or reg['config']['workers']!=workers):raise ValueError('Registration arguments drift')
            if reg['mode']=='full':
                registry=input_root/'runs/phase_d_s2_seed_registry.json'
                with run_lock(registry):
                    book=read_json(registry);entry=book['reservations'][reg['experiment_id']]
                    if entry['status']=='PENDING':
                        entry['status']='COMPLETE';atomic_json(registry,book,overwrite=True)
            return verify_run(output)
        checkpoints=_check_inputs(input_root)
        config=config_for(smoke);config['workers']=workers
        manifest=seed_manifest(smoke)
        temp=output.with_name(output.name+'.registration-pending')
        inventory=dict(ranges=[],unresolved=[],sources={}) if smoke else inventory_seeds(input_root,exclude=(output,temp))
        if inventory['unresolved']:raise ValueError('Unresolved seed history: '+str(inventory['unresolved']))
        for interval in manifest['ranges'].values():
            if any(interval['start']<=b and a<=interval['end_inclusive'] for a,b,_ in inventory['ranges']):
                raise ValueError('Historical seed overlap')
        manifest['inventory_digest']=canonical_digest(inventory)
        reg=dict(format='d-s2-registration-v1',experiment_id='D-S2:'+str(output),mode=manifest['mode'],
            input_root=str(input_root),source_root=str(ROOT),device=device,config=config,checkpoints=checkpoints,
            sources=source_closure(),spec_sha256=sha(ROOT/SPEC),runtime=runtime(),seed_manifest=manifest,
            git={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 'status':subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=ROOT,text=True).splitlines()})
        preflight=ROOT/'docs/superpowers/test-matrices/2026-10-22-phase-d-s2-verification.json'
        reg['preflight']=(dict(evidence=str(preflight),sha256=sha(preflight),measurements=read_json(preflight))
            if preflight.is_file() else dict(status='NOT_MEASURED',note='Run separate smoke before a real experiment.'))
        reg['registration_digest']=canonical_digest(reg)
        temp=output.with_name(output.name+'.registration-pending')
        if temp.exists():
            if not resume:raise FileExistsError('Pending registration exists; use --resume')
            old=read_json(temp/'registration.json')
            if {k:v for k,v in old.items() if k not in ('git','registration_digest')}!={k:v for k,v in reg.items() if k not in ('git','registration_digest')}:
                raise ValueError('Pending registration drift')
            reg=old
        else:
            temp.mkdir(parents=True)
        (temp/'sources').mkdir(exist_ok=True)
        for name,expected in {**reg['sources'],'spec.md':reg['spec_sha256']}.items():
            target=temp/'sources'/name
            if target.exists():
                if sha(target)!=expected:raise ValueError('Prepared source drift: '+name)
            else:
                source=ROOT/SPEC if name=='spec.md' else ROOT/name
                if sha(source)!=expected:raise ValueError('Source changed during registration')
                shutil.copy2(source,target)
        for name,value in [('registration.json',reg),('seed_manifest.json',manifest),('runtime.json',reg['runtime'])]:
            target=temp/name
            if target.exists():
                if read_json(target)!=value:raise ValueError('Prepared metadata drift: '+name)
            else:atomic_json(target,value)
        verify_run(temp,allow_pending=True,check_reservation=False)
        registry=input_root/'runs/phase_d_s2_seed_registry.json'
        if smoke:
            os.replace(temp,output)
        else:
            with run_lock(registry):
                if canonical_digest(inventory_seeds(input_root,exclude=(output,temp)))!=manifest['inventory_digest']:
                    raise ValueError('Seed history changed during registration')
                book=read_json(registry) if registry.exists() else {'reservations':{}}
                own=book['reservations'].get(reg['experiment_id'])
                if own and (not resume or own['registration_digest']!=reg['registration_digest']):
                    raise ValueError('Conflicting existing reservation')
                for key,old in book['reservations'].items():
                    if key==reg['experiment_id']:continue
                    if any(a['start']<=b['end_inclusive'] and b['start']<=a['end_inclusive']
                        for a in manifest['ranges'].values() for b in old['ranges'].values()):raise ValueError('Reserved seed overlap')
                book['reservations'][reg['experiment_id']]=dict(status='PENDING',ranges=manifest['ranges'],
                    registration_digest=reg['registration_digest'],output=str(output))
                atomic_json(registry,book,overwrite=True)
                os.replace(temp,output)
                book['reservations'][reg['experiment_id']]['status']='COMPLETE'
                atomic_json(registry,book,overwrite=True)
        return reg


def verify_run(output,*,allow_pending=False,check_reservation=True):
    output=Path(output).resolve();reg=read_json(output/'registration.json')
    if canonical_digest(reg,'registration_digest')!=reg['registration_digest']:raise ValueError('Registration/config digest drift')
    configure_runtime(reg['device'])
    if runtime()!=reg['runtime']:raise ValueError('Runtime drift')
    if _check_inputs(reg['input_root'])!=reg['checkpoints']:raise ValueError('Checkpoint drift')
    live=Path(reg['source_root'])
    if sha(live/SPEC)!=reg['spec_sha256'] or sha(output/'sources/spec.md')!=reg['spec_sha256']:raise ValueError('Spec drift')
    if read_json(output/'runtime.json')!=reg['runtime'] or read_json(output/'seed_manifest.json')!=reg['seed_manifest']:
        raise ValueError('Runtime/seed manifest drift')
    for name,expected in reg['sources'].items():
        if sha(live/name)!=expected or sha(output/'sources'/name)!=expected or sha(ROOT/name)!=expected:
            raise ValueError('Source drift: '+name)
    if reg['mode']=='full' and check_reservation:
        book=read_json(Path(reg['input_root'])/'runs/phase_d_s2_seed_registry.json')
        entry=book['reservations'].get(reg['experiment_id'],{})
        if entry.get('status') not in (('COMPLETE','PENDING') if allow_pending else ('COMPLETE',)) or entry.get('registration_digest')!=reg['registration_digest']:
            raise ValueError('Incomplete seed reservation')
    return reg


def validate_task_payload(task,payload):
    import math
    canonical_digest(payload)
    for key in ('id','registration_digest','selection_digest'):
        if payload.get(key)!=task.get(key):raise ValueError('Task identity mismatch: '+key)
    if payload.get('format')!='d-s2-board-result-v1':raise ValueError('Task format mismatch')
    rows=payload.get('records',[])
    if len(rows)!=2 or sorted(r.get('corner',-1) for r in rows)!=[0,1]:raise ValueError('Two unique corners required')
    rates=('win','active_win','draw','passive_win','sudden_death_win','suicide','sudden_death_death')
    for row in rows:
        for key,expected in [('board',task['board']),('learner_sha256',task['learner']['sha256']),
            ('opponent_id',task['opponent']['id']),('opponent_sha256',task['opponent']['sha256'])]:
            if row.get(key)!=expected:raise ValueError('Record identity mismatch: '+key)
        if any(row.get(k) not in (0,1) for k in rates):raise ValueError('Invalid binary rate')
        if any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) or row[k]<0
            for k in ('kills','bombs','steps')) or row['steps']>task['max_steps']:raise ValueError('Invalid count/steps')
        if row['active_win']+row['passive_win']+row['sudden_death_win']!=row['win'] or row['win']+row['draw']>1:
            raise ValueError('Win partition mismatch')
        if row['win'] and (row['suicide'] or row['sudden_death_death']):raise ValueError('Dead winner')
        if row['active_win'] and row['kills']<1:raise ValueError('Active win without kill')
        outcome='draw' if row['draw'] else 'loss' if row['win'] else 'win'
        if row.get('opponent_outcome')!=outcome:raise ValueError('Opponent outcome mismatch')
    return rows
