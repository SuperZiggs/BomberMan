"""D1 paired U4/P4 pilot: fixed MC targets, four epochs, a common post-update KL guard."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from audit_t2 import digest, dump
from pilot_two_bomb import stage_timer
from run_combat_move_pilot import (ROOT,WARM_START,B_SHA256,BaseNet,SOURCE_FILES as SHARED_FILES,
    fresh_test_check,train_arm,pair_ready,evaluate_models,comparison_gate,probe_summary)
from run_t3_curriculum import skill_probe
from training_state import runtime_state,run_lock,load_training_state

TRAIN_SEED = 2026101507
TEST_SEED = 37000
ARMS = {'control':'ratio','ppo':'ppo'}
SOURCE_FILES = (*SHARED_FILES,'training_state.py','test_training_state.py','test_ppo_d0.py',
                'test_ppo_d1.py','run_ppo_pilot.py','test_ppo_pilot.py')


def ready(arms):
    return pair_ready(arms,ARMS,max_updates_per_episode=4)


def verify_updates(output,arms):
    """Actual Adam counts, not requested epochs, must agree with committed diagnostics."""
    for arm,objective in ARMS.items():
        folder=output/arm
        rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
        updates=0
        for row in rows:
            d=row['ratio_update']
            assert d['objective']==objective and d['epochs_requested']==4 and 1<=d['epochs']<=4
            assert d['clipped']==(objective=='ppo') and d['target_kl']==.02 and d['clip_epsilon']==.2
            assert len(d['epoch_metrics'])==d['epochs']
            assert [e['epoch'] for e in d['epoch_metrics']]==list(range(1,d['epochs']+1))
            assert all(e['exact_kl_after_update']<=.03 for e in d['epoch_metrics'][:-1])
            assert d['guard_exceeded']==(d['epoch_metrics'][-1]['exact_kl_after_update']>.03)
            assert d['early_stop']==(d['guard_exceeded'] and d['epochs']<4)
            updates+=d['epochs']
            assert row['optimizer_steps']==updates
        state=load_training_state(folder/'training_state.pt')
        assert state['episode']==len(rows)==arms[arm]['completed_episodes']
        assert {int(v['step']) for v in state['optimizer']['state'].values()}=={updates}
        assert state['scheduler']['last_epoch']==len(rows) and updates==arms[arm]['optimizer_steps']


def run(output,device='cpu',smoke=False,budget_minutes=150,resume=False):
    output=Path(output).resolve()
    if not 0<budget_minutes<=150: raise ValueError('Budget must be in (0,150] minutes')
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable')
    if output.exists() and not resume: raise FileExistsError(f'Refusing to overwrite {output}')
    if resume and not (output/'config.json').exists(): raise ValueError('No registered run to resume')
    if digest(WARM_START)!=B_SHA256: raise ValueError('Warm-start B changed')
    with run_lock(output): return _run(output,device,smoke,budget_minutes,resume)


def _run(output,device,smoke,budget_minutes,resume):
    if not smoke and not (resume and (output/'test_opened.json').exists()): fresh_test_check(TEST_SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    sources={name:digest(ROOT/name) for name in SOURCE_FILES}
    config={'registered_utc':datetime.now(timezone.utc).isoformat(),'experiment':'T4-D1',
        'full_protocol':not smoke,'warm_start':str(WARM_START),'warm_start_sha256':B_SHA256,
        'source_sha256':sources,'runtime':runtime_state(),'train_device':device,'arms':ARMS,
        'train_seed':TRAIN_SEED,'episodes_per_arm':20 if smoke else 200,'phase':'C','board_size':9,
        'num_agents':2,'max_steps':20 if smoke else 500,'lr':1e-4,'gamma':.99,'wall_destroy_bonus':.15,
        'attack_teacher':'local','advantage_estimator':'mc','normalize_advantage':False,'critic_target':'MC',
        'update_epochs':4,'clip_epsilon':.2,'target_kl':.02,'kl_threshold':.03,
        'scheduler_cadence':'one step per completed episode','resumable_training_state':True,
        'validation':{'seed':21000,'games':2 if smoke else 100,'every':10 if smoke else 100},
        'evaluation':{'seed':39870 if smoke else TEST_SEED,'games_per_model':2 if smoke else 400,
            'last_board_seed':39870 if smoke else TEST_SEED+199,'both_corners':True,'device':'cpu'},
        'budget_minutes':{'overall':budget_minutes,'preflight':15,'training':100,'evaluation':50},
        'only_treatment_change':'Unclipped U4 versus clipped P4; same four maximum epochs and post-update KL guard',
        'gate':{'gain_vs_both_pp':5,'ci95_lower_above':0,'suicide_not_above_either':True}}
    if resume:
        stored=json.loads((output/'config.json').read_text())
        if {k:v for k,v in stored.items() if k!='registered_utc'}!={k:v for k,v in config.items() if k!='registered_utc'}:
            raise ValueError('Resume source/config/runtime changed')
        config=stored
        if (output/'summary.json').exists() and json.loads((output/'summary.json').read_text())['status']=='complete':
            raise ValueError('Run already complete')
    else:
        output.mkdir(parents=True)
        (output/'source').mkdir()
        for name in SOURCE_FILES: shutil.copy2(ROOT/name,output/'source'/name)
        dump(output/'config.json',config)
    deadline=datetime.fromisoformat(config['registered_utc'])+timedelta(minutes=budget_minutes)
    if datetime.now(timezone.utc)>=deadline: raise ValueError('Original deadline expired; training state is preserved')
    if resume:
        history=output/'resume_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        history.mkdir(parents=True)
        if (output/'summary.json').exists(): shutil.copy2(output/'summary.json',history/'summary.json')
        archive=Path(str(output)+'-send.zip')
        if archive.exists(): shutil.copy2(archive,Path(str(output)+'-before-resume-'+history.name+'.zip'))
    (ROOT/'runs'/('ppo-smoke-last-path.txt' if smoke else 'ppo-last-path.txt')).write_text(str(output),encoding='utf-8')
    started=perf_counter()
    arms,results,selected,comparisons={},{},{},{}
    reason,eligible,timer=None,False,None
    def phase(name,minutes):
        marker=output/f'{name}_deadline.json'
        if marker.exists(): end=datetime.fromisoformat(json.loads(marker.read_text())['deadline_utc'])
        else:
            end=min(deadline,datetime.now(timezone.utc)+timedelta(minutes=minutes))
            dump(marker,{'deadline_utc':end.isoformat()})
        return stage_timer(max(0,(min(end,deadline)-datetime.now(timezone.utc)).total_seconds()))
    try:
        stop,timer=(stage_timer(max(0,(deadline-datetime.now(timezone.utc)).total_seconds()))
                    if resume else phase('preflight',15))
        net=BaseNet(board_size=9)
        import train_actor_critic as trainer
        trainer.smart_load_weights(net,str(WARM_START),'cpu')
        baseline=skill_probe(net,1,stop,smoke=False)
        if baseline is None: reason='baseline_skill_incomplete'
        else:
            dump(output/'baseline_skill.json',probe_summary(baseline))
            timer.cancel()
            stop,timer=phase('training',100)
            for arm,objective in ARMS.items():
                saved=output/arm/'summary.json'
                if resume and saved.exists():
                    cached=json.loads(saved.read_text())
                    if pair_ready({arm:cached},(arm,),max_updates_per_episode=4):
                        for item in (*cached['checkpoints'].values(),cached['training_state']):
                            if digest(Path(item['path']))!=item['sha256']: raise ValueError('Completed arm checkpoint changed')
                        arms[arm]=cached
                        continue
                if stop.is_set(): break
                arms[arm]=train_arm(output,arm,device,smoke,stop,baseline,attack_teacher='local',
                    normalize_advantage=False,advantage_estimator='mc',train_seed=TRAIN_SEED,
                    reward_diagnostics=True,contact_diagnostics=True,save_training_state=True,
                    resume=resume and (output/arm).exists(),smoke_episodes=20,smoke_steps=20,
                    policy_objective=objective,update_epochs=4,clip_epsilon=.2,target_kl=.02)
                if not pair_ready({arm:arms[arm]},(arm,),max_updates_per_episode=4): break
            if not ready(arms): reason='training_validation_or_skill_incomplete'
            else:
                verify_updates(output,arms)
                assert sources=={name:digest(ROOT/name) for name in SOURCE_FILES} and digest(WARM_START)==B_SHA256
                selected={'original':{'path':str(WARM_START),'sha256':B_SHA256},
                    **{arm:arms[arm]['checkpoints']['best'] for arm in ARMS}}
                if (output/'test_opened.json').exists():
                    if json.loads((output/'selected_models.json').read_text())!=selected:
                        raise ValueError('Test models locked; reselection forbidden')
                else:
                    dump(output/'selected_models.json',selected)
                    dump(output/'test_opened.json',{'seed':39870 if smoke else TEST_SEED,
                        'num_episodes':2 if smoke else 400,'num_agents':2})
                timer.cancel()
                stop,timer=phase('evaluation',50)
                results=evaluate_models(output,stop,smoke,arm_names=ARMS,test_seed=TEST_SEED,
                    smoke_seed=39870,contact_diagnostics=True,resume=resume)
                if len(results)!=3: reason='evaluation_incomplete'
                elif not smoke: comparisons,eligible=comparison_gate(results,'ppo')
    except KeyboardInterrupt: reason='interrupted'
    except Exception as error:
        reason=f'{type(error).__name__}: {error}'
        raise
    finally:
        if timer is not None: timer.cancel()
        unchanged=digest(WARM_START)==B_SHA256 and sources=={name:digest(ROOT/name) for name in SOURCE_FILES}
        checkpoint_ok=all(digest(Path(r['path']))==r['sha256'] for r in selected.values())
        complete=reason is None and ready(arms) and len(results)==3 and unchanged and checkpoint_ok
        summary={'status':'complete' if complete else 'incomplete','full_protocol':not smoke,'reason':reason,
            'arms':arms,'evaluation_complete':len(results)==3,'hashes_unchanged':unchanged,
            'selected_checkpoints_unchanged':checkpoint_ok,'comparisons':comparisons,
            'eligible_for_replication':complete and not smoke and eligible,
            'decision':'replicate_before_replacing' if complete and not smoke and eligible else 'retain_original',
            'elapsed_seconds':(datetime.now(timezone.utc)-datetime.fromisoformat(config['registered_utc'])).total_seconds(),
            'resumed':resume,'models':{name:{k:v for k,v in r.items() if k!='games'} for name,r in results.items()}}
        dump(output/'summary.json',summary)
        shutil.make_archive(str(output)+'-send','zip',root_dir=output)
        print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--budget-minutes',type=float,default=150)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs'/('ppo-full-'+datetime.now().strftime('%Y%m%d-%H%M%S')))
    args=parser.parse_args()
    if run(args.output_dir,args.device,args.smoke,args.budget_minutes,args.resume)['status']!='complete': raise SystemExit(1)
