"""D-S2 episode control and persistence; training math lives in the shared trainer."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil

from phase_d_s2_protocol import (CHECKPOINTS, ROOT, atomic_json, config_for, read_json,
                               sha, verify_run)

FORMAT = 'bomberman_phase_d_training_v1'


def build_train_schedule(arm, *, snapshots=(), recent_cursor=0, start_episode=1,
                         episodes=2000, smoke=False):
    if arm not in ('control','pool') or start_episode<1 or episodes<0 or recent_cursor<0:
        raise ValueError('Invalid arm/schedule bounds')
    rows=[];pairs={};cursor=recent_cursor
    slots=config_for(smoke)['pool_slots']
    for ep in range(start_episode,start_episode+episodes):
        pair=(ep-1)//2;block=(ep-1)//20;slot=pair%10
        if pair not in pairs:
            name='RB' if arm=='control' else slots[slot]
            typ='RB' if name=='RB' else 'recent' if name=='recent' else 'fixed'
            fallback=False
            if name=='recent':
                eligible=sorted((s for s in snapshots if s.get('arm')=='pool' and
                    isinstance(s.get('episode'),int) and s['episode']<block*20+1),key=lambda s:s['episode'])[-2:]
                chosen=eligible[cursor%len(eligible)] if eligible else None
                cursor+=1
                name=chosen['id'] if chosen else 'S2';fallback=chosen is None
                digest=chosen['sha256'] if chosen else CHECKPOINTS['S2'][1]
            else:
                digest=sha(ROOT/'rule_based_bot.py') if name=='RB' else CHECKPOINTS[name][1]
            pairs[pair]=dict(block=block,slot=slot,board_seed=(690000 if smoke else 610000)+pair,
                opponent_type=typ,opponent_id=name,opponent_sha256=digest,recent_fallback=fallback)
        rows.append(dict(episode=ep,learner_corner=(ep-1)%2,**pairs[pair]))
    return dict(episodes=rows,recent_cursor=cursor)


def tensor_digest(net):
    h=hashlib.sha256()
    for key,value in net.state_dict().items():
        h.update(key.encode());h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class PhaseDRun:
    def __init__(self,output,arm,registration,commit_hook=None):
        self.output=Path(output);self.arm=arm;self.registration=registration
        self.directory=self.output/'arms'/arm;self.commit_hook=commit_hook
        self.smoke=registration['mode']=='smoke';self.config=registration['config']
        self.state=dict(recent_cursor=0,pair=None,snapshots=[],candidates=[],metrics=[],opponent_audit=[])
        self.cache={};self.audit={};self.last_audit=None

    def _opponent(self,desc,device):
        import torch
        from train_actor_critic import ActorCriticNet,SnapshotOpponent,smart_load_weights
        key=desc['sha256']
        if key not in self.cache:
            # Loading a frozen network must not advance learner sampling RNG on resume.
            with torch.random.fork_rng(devices=[]):
                net=ActorCriticNet(in_channels=13,board_size=9).to(device)
                smart_load_weights(net,desc['checkpoint'],device)
            net.eval().requires_grad_(False)
            if len(self.cache)>=2:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key]=SnapshotOpponent(net,device)
            digest=tensor_digest(net)
            self.audit[desc['id']]=dict(id=desc['id'],before_sha256=digest,after_sha256=digest,
                eval_mode=True,gradient_count=0)
        opp=self.cache[key]
        row=self.audit.setdefault(desc['id'],dict(id=desc['id'],before_sha256=tensor_digest(opp.net)))
        row.update(after_sha256=tensor_digest(opp.net),eval_mode=not opp.net.training,
            gradient_count=sum(p.grad is not None for p in opp.net.parameters()))
        if row['before_sha256']!=row['after_sha256'] or row['gradient_count'] or not row['eval_mode']:
            raise ValueError('Frozen opponent was modified')
        return opp

    def prepare_episode(self,ep,env,device):
        import numpy as np
        pair=(ep-1)//2
        saved=self.state['pair']
        if saved is not None and (saved['episode']-1)//2==pair:
            audit={**saved,'episode':ep,'learner_corner':(ep-1)%2}
        else:
            result=build_train_schedule(self.arm,snapshots=self.state['snapshots'],
                recent_cursor=self.state['recent_cursor'],start_episode=ep,episodes=1,smoke=self.smoke)
            audit=result['episodes'][0];self.state['recent_cursor']=result['recent_cursor']
            self.state['pair']=audit
        self.last_audit=audit
        env.rng=np.random.default_rng(audit['board_seed'])
        name=audit['opponent_id'];aid=audit['learner_corner']
        if name=='RB':opp='rule'
        else:
            desc=self.registration['checkpoints'].get(name)
            if desc is None:desc=next(s for s in self.state['snapshots'] if s['id']==name)
            opp=self._opponent(desc,device)
        return dict(learning_aid=aid,opponents={1-aid:opp},audit=audit)

    def snapshot_completed(self,ep,net):
        from training_state import atomic_save
        weights={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        for kind,enabled in [('snapshots',ep>0 and ep%self.config['train']['snapshot_every']==0),
                             ('candidates',ep in self.config['candidate_episodes'])]:
            if not enabled:continue
            path=self.directory/('checkpoints' if kind=='candidates' else kind)/f'episode_{ep}.pt'
            if ep==0:
                path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(self.registration['checkpoints']['S2']['checkpoint'],path)
            else:atomic_save(weights,path)
            desc=dict(id=f'snapshot-{ep}' if kind=='snapshots' else f'{self.arm}-{ep}',
                checkpoint=str(path.resolve()),sha256=sha(path),episode=ep,role='candidate' if kind=='candidates' else 'opponent',arm=self.arm)
            self.state[kind]=[s for s in self.state[kind] if s['episode']!=ep]+[desc]

    def record_episode(self,row):
        row={**row,**self.last_audit,'optimizer_steps':row['episode']}
        self.state['metrics'].append(row)
        return row

    def state_dict(self):
        self.state['opponent_audit']=list(self.audit.values())
        return deepcopy(self.state)

    def load_state_dict(self,state):
        self.state=deepcopy(state);self.cache={}
        for desc in self.state['snapshots']+self.state['candidates']:
            path=Path(desc['checkpoint']).resolve()
            if not path.is_relative_to(self.directory.resolve()) or sha(path)!=desc['sha256']:
                raise ValueError('Snapshot/candidate path or hash mismatch')
        self.audit={r['id']:r for r in state['opponent_audit']}

    def after_commit(self,ep):
        atomic_json(self.directory/'opponent_manifest.json',{'snapshots':self.state['snapshots'],
            'candidates':self.state['candidates']},overwrite=True)
        path=self.directory/'metrics.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.jsonl.tmp')
        with temporary.open('w',encoding='utf-8') as f:
            for row in self.state['metrics']:f.write(json.dumps(row,allow_nan=False)+'\n')
            f.flush();__import__('os').fsync(f.fileno())
        __import__('os').replace(temporary,path)


def train_arm(output,arm,*,resume=False,stop_event=None,on_episode=None,on_step=None,commit_hook=None):
    import numpy as np
    import torch
    from progress import bar
    from training_state import run_lock,load_training_state
    from train_actor_critic import train
    if arm not in ('control','pool'):raise ValueError('Unknown arm')
    output=Path(output).resolve()
    with run_lock(output):
        reg=verify_run(output);directory=output/'arms'/arm;state_path=directory/'training_state.pt'
        if directory.exists() and not resume:raise FileExistsError('Arm exists; use --resume')
        if resume and not state_path.is_file():raise ValueError('Missing training state')
        directory.mkdir(parents=True,exist_ok=resume)
        context=PhaseDRun(output,arm,reg,commit_hook)
        done=load_training_state(state_path,expected_format=FORMAT)['episode'] if resume else 0
        seed=reg['config']['train']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if reg['device'].startswith('cuda'):torch.cuda.manual_seed_all(seed)
        with bar(reg['config']['train']['episodes'],f'D-S2 {arm}',done) as shown:
            def episode(row):
                shown.update(1)
                if on_episode:on_episode(row)
            train(**reg['config']['train'],device=reg['device'],checkpoint_dir=str(directory),
                resume_from=reg['checkpoints']['S2']['checkpoint'],save_training_state=True,
                resume_training_state=str(state_path) if resume else '',phase_d_run=context,
                on_episode=episode,on_step=on_step,stop_event=stop_event)
        state=load_training_state(state_path,expected_format=FORMAT)
        context.load_state_dict(state['context']);context.after_commit(state['episode'])
        if state['episode']==reg['config']['train']['episodes']:
            atomic_json(directory/'completed.json',dict(arm=arm,episode=state['episode'],
                registration_digest=reg['registration_digest'],state_sha256=sha(state_path)),overwrite=True)


def read_training_state(output,arm):
    from training_state import load_training_state
    state=load_training_state(Path(output)/'arms'/arm/'training_state.pt',expected_format=FORMAT)
    context=state['context']
    return dict(**{k:state[k] for k in ('episode','model','optimizer','scheduler','rng')},
        schedule_state={k:context[k] for k in ('recent_cursor','pair')},metrics=context['metrics'],
        snapshots=context['snapshots'],initial_sha256=state['provenance']['initial_sha256'],
        initial_optimizer_steps=0,opponent_audit=context['opponent_audit'])
