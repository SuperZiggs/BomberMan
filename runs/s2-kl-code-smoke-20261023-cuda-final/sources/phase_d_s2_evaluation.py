"""Board-cluster evaluation, selection and pre-registered D-S2 decisions."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path
import random
import time

import numpy as np

from phase_d_s2_protocol import (CNN, ROOT, canonical_digest, read_json, sha,
    configure_runtime, validate_task_payload, verify_run)

RATES=('win','active_win','draw','passive_win','sudden_death_win','suicide','sudden_death_death')


def load_registered_model(checkpoint,model_id,device='cpu'):
    import torch
    from train_actor_critic import ActorCriticNet
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not isinstance(weights,dict) or 'conv.0.weight' not in weights:raise ValueError('Invalid inference checkpoint')
    conv=weights['conv.0.weight']
    expected=12 if model_id=='B' else 13
    if not torch.is_tensor(conv) or conv.ndim!=4 or conv.shape[1]!=expected:
        raise ValueError('Unsupported input channels for '+model_id)
    with torch.random.fork_rng(devices=[]):net=ActorCriticNet(in_channels=13,board_size=9)
    reference=net.state_dict()
    if set(weights)!=set(reference):raise ValueError('Checkpoint keys mismatch')
    weights=dict(weights)
    if model_id=='B':
        if tuple(conv.shape)!=(32,12,3,3):raise ValueError('Unsupported B convolution shape')
        weights['conv.0.weight']=torch.cat((conv,torch.zeros_like(conv[:,:1])),dim=1)
    if any(not torch.is_tensor(v) or v.shape!=reference[k].shape for k,v in weights.items()):
        raise ValueError('Checkpoint tensor/9x9 board shape mismatch')
    net.load_state_dict(weights,strict=True)
    return net.to(device).eval().requires_grad_(False)


def _macro(candidate,field):
    return sum(float(candidate['metrics'][o][field]) for o in CNN)/len(CNN)


def select_candidates(candidates):
    baseline=deepcopy(candidates['S2']);baseline['episode']=0
    for opp in (*CNN,'RB'):
        if opp not in baseline.get('metrics',{}):raise ValueError('Incomplete baseline validation')
    def guard(c):
        try:
            for opp in (*CNN,'RB'):
                for k in ('active_win_rate','win_rate','draw_rate','suicide_rate'):
                    v=c['metrics'][opp][k]
                    if not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=100:raise ValueError('Invalid rate')
            return (c['metrics']['RB']['active_win_rate']>=baseline['metrics']['RB']['active_win_rate']-2
                and c['metrics']['RB']['suicide_rate']<=baseline['metrics']['RB']['suicide_rate']+1
                and _macro(c,'suicide_rate')<=_macro(baseline,'suicide_rate')+1)
        except KeyError as e:raise ValueError('Incomplete candidate validation') from e
    def score(c):
        return (_macro(c,'active_win_rate'),-_macro(c,'draw_rate'),_macro(c,'win_rate'),
                -_macro(c,'suicide_rate'),-c['episode'])
    def better(a,b):
        for x,y in zip(score(a),score(b)):
            if abs(x-y)>1e-12:return x>y
        return False
    models={'S2':baseline};summary={}
    for arm in ('control','pool'):
        selected=baseline
        checks=[]
        for c in candidates.get(arm,[]):
            accepted=guard(c);checks.append(dict(id=c['id'],guard=accepted,score=list(score(c))))
            if accepted and better(c,selected):selected=c
        models[arm]=deepcopy(selected);summary[arm]=checks
    delta=_macro(models['pool'],'active_win_rate')-_macro(baseline,'active_win_rate')
    opened=models['pool']['episode']>0 and delta>=3 and (
        _macro(models['pool'],'active_win_rate')>=_macro(models['control'],'active_win_rate'))
    return dict(models=models,validation_summary=summary,open_test=opened)


def paired_macro_comparison(left_rows,right_rows,metric,opponents=CNN):
    if not opponents or len(set(opponents))!=len(opponents):raise ValueError('Invalid opponent set')
    expected={(o,c) for o in opponents for c in (0,1)}
    def boards(rows):
        result=defaultdict(dict)
        for row in rows:
            if row['opponent_id'] not in opponents:continue
            key=(row['opponent_id'],row['corner']);board=row['board']
            if key in result[board]:raise ValueError('Duplicate board corner/opponent')
            v=row[metric]
            if not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('Invalid bootstrap metric')
            result[board][key]=v
        if not result or any(set(v)!=expected for v in result.values()):raise ValueError('Incomplete paired boards')
        return result
    a,b=boards(left_rows),boards(right_rows)
    if set(a)!=set(b):raise ValueError('Misaligned paired boards')
    deltas=np.array([np.mean(list(a[k].values()))-np.mean(list(b[k].values())) for k in sorted(a)])
    samples=np.random.default_rng(20261014).choice(deltas,size=(10000,len(deltas)),replace=True).mean(axis=1)
    return dict(difference_pp=float(deltas.mean()*100),ci95_pp=[float(v) for v in np.quantile(samples,[.025,.975])*100],
                board_pairs=len(deltas),resamples=10000,bootstrap_seed=20261014)


def macro_board_bootstrap(left,right,opponents=CNN):
    r=paired_macro_comparison(left,right,'active_win',opponents)
    return dict(r,ci95_lower_pp=r['ci95_pp'][0],ci95_upper_pp=r['ci95_pp'][1])


def decide_gates(comparisons,pool_matchups,baseline_rb,snapshot_ids):
    conditions={}
    try:
        p=comparisons['pool_minus_S2'];c=comparisons['pool_minus_control']['macro_cnn']
        conditions=dict(active_gain=p['macro_cnn']['active_win']['difference_pp']>=5,
            active_ci=p['macro_cnn']['active_win']['ci95_pp'][0]>0,
            control_ci=c['active_win']['ci95_pp'][0]>0,
            draw=p['macro_cnn']['draw']['difference_pp']<=0,
            suicide=p['macro_cnn']['suicide']['difference_pp']<=1,
            rb_active=p['RB']['active_win']['difference_pp']>=-2,
            rb_suicide=p['RB']['suicide']['difference_pp']<=1)
        if any(not math.isfinite(float(v)) for x in (p['macro_cnn'],p['RB'],c)
               for item in x.values() for v in [item['difference_pp'],*item['ci95_pp']]):raise ValueError('Nonfinite comparison')
    except (KeyError,IndexError,TypeError,ValueError):conditions={'complete_comparisons':False}
    pilot=all(conditions.values())
    checks={'pilot':pilot,'has_old_snapshots':bool(snapshot_ids)}
    for name in (*CNN,*snapshot_ids):
        m=pool_matchups.get(name,{})
        checks[name]=m.get('games')==200 and m.get('win_rate',-1)>=60
    rb=pool_matchups.get('RB',{})
    checks['RB']=rb.get('games')==200 and baseline_rb.get('games')==200 and rb.get('win_rate',-1)>=baseline_rb.get('win_rate',101)
    phase=all(checks.values())
    return dict(pilot_gate=dict(pass_=pilot,status='PILOT_PASS' if pilot else 'PILOT_NOT_PASS',conditions=conditions,**{'pass':pilot}),
        phase_d_gate=dict(status='PHASE_D_PASS' if phase else 'PHASE_D_NOT_PASS',conditions=checks,**{'pass':phase}))


def _reference(reg,name):
    if name=='RB' and name not in reg['checkpoints']:
        return dict(id='RB',checkpoint=None,sha256=reg['sources']['rule_based_bot.py'],episode=None,role='anchor')
    return deepcopy(reg['checkpoints'][name])


def _tasks(reg,stage,learners,opponents,range_name):
    tasks=[];span=reg['seed_manifest']['ranges'][range_name]
    unique={}
    for desc in learners:unique.setdefault(desc['sha256'],desc)
    for learner in unique.values():
        for opponent in opponents:
            for board in range(span['start'],span['end_inclusive']+1):
                identity=dict(stage=stage,learner=learner['sha256'],opponent=opponent['id'],
                              opponent_sha256=opponent['sha256'],board=board)
                tasks.append(dict(id=canonical_digest(identity),stage=stage,learner=deepcopy(learner),
                    opponent=deepcopy(opponent),board=board,corners=[0,1],max_steps=reg['config']['train']['max_steps'],
                    registration_digest=reg['registration_digest'],selection_digest=None))
    return tasks


def build_test_manifest(registration,models,snapshots):
    reg=registration
    tasks=_tasks(reg,'test',models.values(),[_reference(reg,o) for o in (*CNN,'B','RB')],'test_reference')
    selected=models['pool'];seen=set()
    for snap in sorted(snapshots,key=lambda s:s['episode']):
        if snap.get('arm')!='pool' or snap['episode']>=selected['episode'] or snap['sha256']==selected['sha256']:continue
        if snap['id'] in seen:raise ValueError('Duplicate snapshot identity')
        seen.add(snap['id'])
        tasks.extend(_tasks(reg,'test',[selected],[snap],'test_snapshots'))
    return tasks


def candidates_for_run(output,reg):
    from training_state import load_training_state
    result={'S2':dict(reg['checkpoints']['S2'],episode=0)}
    snapshots=[]
    for arm in ('control','pool'):
        directory=Path(output)/'arms'/arm;marker=read_json(directory/'completed.json')
        state_path=directory/'training_state.pt'
        if (marker['episode']!=reg['config']['train']['episodes'] or marker['state_sha256']!=sha(state_path)
            or marker['registration_digest']!=reg['registration_digest']):raise ValueError('Incomplete/drifted training arm')
        state=load_training_state(state_path,expected_format='bomberman_phase_d_training_v1')
        if state['episode']!=marker['episode'] or state['provenance'].get('registration_digest')!=reg['registration_digest']:
            raise ValueError('Training state identity mismatch')
        context=state['context'];result[arm]=context['candidates']
        if arm=='pool':snapshots=context['snapshots']
        for desc in context['candidates']+context['snapshots']:
            path=Path(desc['checkpoint']).resolve()
            if not path.is_relative_to(directory.resolve()) or sha(path)!=desc['sha256']:raise ValueError('Model artifact drift')
    return result,snapshots


def validation_tasks(output):
    reg=verify_run(output);candidates,_=candidates_for_run(output,reg)
    models=[candidates['S2'],*candidates['control'],*candidates['pool']]
    return _tasks(reg,'validation',models,[_reference(reg,o) for o in (*CNN,'RB')],'validation')


def test_tasks(output,selection):
    reg=verify_run(output);_,snaps=candidates_for_run(output,reg)
    return build_test_manifest(reg,selection['models'],snaps)


_POLICIES={}


def _policy(desc):
    import torch
    from train_actor_critic import SnapshotOpponent
    from evaluate import make_rule_based_fn
    if desc['id']=='RB':return make_rule_based_fn(0)
    if sha(desc['checkpoint'])!=desc['sha256']:raise ValueError('Evaluation checkpoint drift')
    if desc['sha256'] not in _POLICIES:
        if len(_POLICIES)>=4:_POLICIES.pop(next(iter(_POLICIES)))
        net=load_registered_model(desc['checkpoint'],desc['id'])
        policy=SnapshotOpponent(net,'cpu')
        _POLICIES[desc['sha256']]=lambda env,obs,aid:policy.select_action(obs[aid],env,aid)
    return _POLICIES[desc['sha256']]


def evaluate_board(task):
    import torch
    from bomberman_env import BombermanFFAEnv
    from evaluate import run_match
    configure_runtime('cpu')
    state=random.getstate();npstate=np.random.get_state();rows=[]
    try:
        with torch.random.fork_rng(devices=[]):
            for corner in task['corners']:
                started=time.perf_counter()
                board=task['board'];random.seed(board);np.random.seed(board);torch.manual_seed(board)
                env=BombermanFFAEnv(board_size=9,num_agents=2,max_steps=task['max_steps'],seed=board,
                    sudden_death=True,sudden_death_start=475,sudden_death_interval=25,sudden_death_warning_steps=40)
                policies=[None,None];policies[corner]=_policy(task['learner']);policies[1-corner]=_policy(task['opponent'])
                stats=run_match(env,policies,num_episodes=1)
                row=dict(learner_sha256=task['learner']['sha256'],opponent_id=task['opponent']['id'],
                    opponent_sha256=task['opponent']['sha256'],board=board,corner=corner,
                    win=stats['wins'][corner],active_win=stats['active_wins'][corner],draw=stats['draws'],
                    passive_win=stats['passive_wins'][corner],sudden_death_win=stats['sudden_death_wins'][corner],
                    suicide=stats['suicide_rates'][corner]/100,sudden_death_death=stats['sudden_death_death_rates'][corner]/100,
                    kills=stats['avg_kills'][corner],bombs=stats['avg_bombs'][corner],steps=stats['avg_steps'])
                row['opponent_outcome']='draw' if row['draw'] else 'loss' if row['win'] else 'win'
                row['elapsed_seconds']=time.perf_counter()-started
                rows.append(row)
    finally:random.setstate(state);np.random.set_state(npstate)
    payload=dict(format='d-s2-board-result-v1',id=task['id'],registration_digest=task['registration_digest'],
        selection_digest=task['selection_digest'],records=rows,
        worker_runtime=dict(device='cpu',threads=torch.get_num_threads(),
            deterministic=torch.are_deterministic_algorithms_enabled(),source_root=str(ROOT)))
    validate_task_payload(task,payload)
    return payload


validate_task=validate_task_payload


def summarize(rows):
    if not rows:raise ValueError('Empty matchup')
    return dict(games=len(rows),**{k+'_rate':float(np.mean([r[k] for r in rows])*100) for k in RATES},
        elapsed_seconds=sum(r.get('elapsed_seconds',0) for r in rows),
        by_corner={str(c):dict(games=sum(r['corner']==c for r in rows),
            **{k+'_rate':float(np.mean([r[k] for r in rows if r['corner']==c])*100) for k in RATES})
            for c in (0,1) if any(r['corner']==c for r in rows)},
        **{'avg_'+k:float(np.mean([r[k] for r in rows])) for k in ('kills','bombs','steps')})


def select_models(rows,candidates):
    enriched=deepcopy(candidates)
    for desc in [enriched['S2'],*enriched['control'],*enriched['pool']]:
        own=[r for r in rows if r['learner_sha256']==desc['sha256']]
        desc['metrics']={o:summarize([r for r in own if r['opponent_id']==o]) for o in (*CNN,'RB')}
    return select_candidates(enriched)


def build_report(rows,selection):
    models=selection['models'];summary={};own={}
    for name,desc in models.items():
        own[name]=[r for r in rows if r['learner_sha256']==desc['sha256']]
        summary[name]={o:summarize([r for r in own[name] if r['opponent_id']==o])
            for o in sorted({r['opponent_id'] for r in own[name]})}
    comparisons={}
    for right in ('S2','control'):
        comparisons['pool_minus_'+right]={'macro_cnn':{m:paired_macro_comparison(own['pool'],own[right],m)
            for m in ('active_win','win','draw','suicide')}}
        for opponent in ('RB','B'):
            comparisons['pool_minus_'+right][opponent]={m:paired_macro_comparison(own['pool'],own[right],m,(opponent,))
                for m in ('active_win','win','draw','suicide')}
    snapshots=sorted(set(summary['pool'])-set((*CNN,'B','RB')))
    gates=decide_gates(comparisons,summary['pool'],summary['S2']['RB'],snapshots)
    if selection.get('mode')=='smoke':
        gates['phase_d_gate'].update({'pass':False,'status':'SMOKE_ONLY'})
    status=gates['phase_d_gate']['status'] if gates['phase_d_gate']['pass'] else gates['pilot_gate']['status']
    if selection.get('mode')=='smoke':status='SMOKE_ONLY'
    return dict(status=status,complete=True,actual_games=len(rows),expected_games=2*len(selection['test_manifest']),
        summary=summary,comparisons=comparisons,**gates,
        limitations=['One training seed; fixed known CNN pool; B is outside this run training/selection.',
                     'CIs resample boards, not independent training runs.'])
