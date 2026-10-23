"""Paired RuleBot training through the existing trainer's full-state context."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil

from s2_kl_protocol import (ROOT, S2_SHA, atomic_json, canonical_digest, config_for,
    read_json, seed_manifest, sha, verify_run)

FORMAT = 'bomberman_phase_d_training_v1'


def build_train_schedule(arm, *, smoke=False):
    if arm not in ('control', 'anchor'):
        raise ValueError('Unknown arm')
    first = seed_manifest(smoke)['ranges']['train']['start']
    return [dict(episode=ep, board_seed=first + (ep - 1) // 2,
        learner_corner=(ep - 1) % 2, opponent_id='RB')
        for ep in range(1, config_for(smoke)['train']['episodes'] + 1)]


class S2KLRun:
    def __init__(self, output, arm, registration, commit_hook=None):
        self.output, self.arm, self.registration = Path(output), arm, registration
        self.directory = self.output / 'arms' / arm
        self.commit_hook = commit_hook
        self.schedule = build_train_schedule(arm, smoke=registration['mode'] == 'smoke')
        self.identity = dict(arm=arm, registration_digest=registration['registration_digest'],
            initial_sha256=S2_SHA, reference_sha256=S2_SHA,
            anchor_coef=registration['config']['anchor_coef'][arm])
        self.state = dict(identity=deepcopy(self.identity), cursor=0, candidates=[], metrics=[])
        self.last_audit = None

    def prepare_episode(self, ep, env, device):
        import numpy as np
        if ep != self.state['cursor'] + 1 or not 1 <= ep <= len(self.schedule):
            raise ValueError('Training cursor drift')
        audit = dict(self.schedule[ep - 1], opponent_sha256=self.registration['sources']['rule_based_bot.py'])
        self.last_audit = audit
        env.rng = np.random.default_rng(audit['board_seed'])
        aid = audit['learner_corner']
        return dict(learning_aid=aid, opponents={1 - aid: 'rule'}, audit=audit)

    def snapshot_completed(self, ep, net):
        from training_state import atomic_save
        if ep not in self.registration['config']['checkpoint_episodes']:
            return
        path = self.directory / 'checkpoints' / f'{ep}.pt'
        if ep == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.pt.tmp')
            shutil.copy2(self.registration['checkpoints']['S2']['checkpoint'], temporary)
            with temporary.open('r+b') as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        else:
            atomic_save({k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, path)
        desc = dict(id=f'{self.arm}-{ep}', arm=self.arm, episode=ep,
            checkpoint=str(path.resolve()), sha256=sha(path))
        self.state['candidates'] = [d for d in self.state['candidates'] if d['episode'] != ep] + [desc]

    def record_episode(self, row):
        row = dict(row, **{k: v for k, v in self.last_audit.items() if k not in row},
            optimizer_steps=row['episode'], anchor_enabled=bool(self.identity['anchor_coef']))
        if row['learning_aid'] != self.last_audit['learner_corner']:
            raise ValueError('Training corner drift')
        self.state['metrics'].append(row)
        self.state['cursor'] = row['episode']
        return row

    def state_dict(self):
        return deepcopy(self.state)

    def load_state_dict(self, state):
        if state.get('identity') != self.identity:
            raise ValueError('Arm/S2/reference/KL identity drift')
        cursor = state['cursor']
        if type(cursor) is not int or not 0 <= cursor <= len(self.schedule) or len(state['metrics']) != cursor:
            raise ValueError('Training cursor/history drift')
        expected = [e for e in self.registration['config']['checkpoint_episodes'] if e <= cursor]
        if [d['episode'] for d in state['candidates']] != expected:
            raise ValueError('Candidate inventory drift')
        for desc in state['candidates']:
            path = Path(desc['checkpoint']).resolve()
            if (path != (self.directory / 'checkpoints' / f"{desc['episode']}.pt").resolve() or
                    desc['arm'] != self.arm or desc['id'] != f"{self.arm}-{desc['episode']}" or
                    sha(path) != desc['sha256'] or (desc['episode'] == 0 and desc['sha256'] != S2_SHA)):
                raise ValueError('Candidate path/hash drift')
        for row, expected_row in zip(state['metrics'], self.schedule):
            if any(row.get(k) != v for k, v in expected_row.items()) or row.get('optimizer_steps') != row['episode']:
                raise ValueError('Committed training schedule drift')
        self.state = deepcopy(state)

    def after_commit(self, ep):
        if ep != self.state['cursor']:
            raise ValueError('Projection cursor drift')
        path = self.directory / 'metrics.jsonl'
        existing = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []
        if existing != self.state['metrics'][:len(existing)] or len(existing) > ep:
            raise ValueError('Training metrics projection drift')
        text = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in self.state['metrics'])
        if path.exists() and path.read_text(encoding='utf-8') == text:
            return
        temporary = path.with_suffix('.jsonl.tmp')
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def train_arm(output, arm, *, resume=False, stop_event=None, commit_hook=None):
    import numpy as np
    import torch
    from progress import bar
    from training_state import load_training_state, run_lock
    from train_actor_critic import train
    if arm not in ('control', 'anchor'):
        raise ValueError('Unknown arm')
    output = Path(output).resolve()
    with run_lock(output):
        reg = verify_run(output)
        directory = output / 'arms' / arm
        state_path = directory / 'training_state.pt'
        if directory.exists() and not resume:
            raise FileExistsError('Arm exists; use --resume')
        marker = directory / 'initializing.json'
        identity = dict(format='s2-kl-arm-initialization-v1', arm=arm,
            registration_digest=reg['registration_digest'], initial_sha256=S2_SHA,
            reference_sha256=S2_SHA, anchor_coef=reg['config']['anchor_coef'][arm])
        has_state = state_path.is_file()
        if resume and not has_state:
            if not marker.is_file() or read_json(marker) != identity:
                raise ValueError('Missing training state/initialization identity')
            files = [p for p in directory.rglob('*') if p.is_file() and p.suffix != '.tmp']
            allowed = {marker.resolve(), (directory / 'checkpoints/0.pt').resolve()}
            if any(p.resolve() not in allowed for p in files):
                raise ValueError('Committed artifacts exist without training state')
            initial = directory / 'checkpoints/0.pt'
            if initial.exists() and sha(initial) != S2_SHA:
                raise ValueError('Initial checkpoint drift')
        directory.mkdir(parents=True, exist_ok=resume)
        if marker.exists():
            if read_json(marker) != identity:
                raise ValueError('Arm initialization drift')
        else:
            atomic_json(marker, identity)
        context = S2KLRun(output, arm, reg, commit_hook)
        done = 0
        if has_state:
            state = load_training_state(state_path, expected_format=FORMAT)
            context.load_state_dict(state['context'])
            if state['episode'] != context.state['cursor']:
                raise ValueError('State/context cursor drift')
            done = state['episode']
            context.after_commit(done)
        seed = reg['config']['train']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if reg['device'].startswith('cuda'):
            torch.cuda.manual_seed_all(seed)
        target = reg['config']['train']['episodes']
        with bar(target, 'S2-KL ' + arm, done) as shown:
            train(**reg['config']['train'], anchor_coef=reg['config']['anchor_coef'][arm],
                device=reg['device'], checkpoint_dir=str(directory),
                resume_from=reg['checkpoints']['S2']['checkpoint'], save_training_state=True,
                resume_training_state=str(state_path) if has_state else '', phase_d_run=context,
                on_episode=lambda row: shown.update(1), stop_event=stop_event)
        state = load_training_state(state_path, expected_format=FORMAT)
        context.load_state_dict(state['context'])
        context.after_commit(state['episode'])
        complete = state['episode'] == target
        if complete:
            atomic_json(directory / 'completed.json', dict(arm=arm, episode=target,
                registration_digest=reg['registration_digest'], state_sha256=sha(state_path)), overwrite=True)
        return dict(arm=arm, episode=state['episode'], target=target, complete=complete)


def model_manifest(reg, descriptors):
    unique = {}
    for desc in descriptors:
        unique.setdefault(desc['sha256'], desc)
    endpoint = reg['config']['endpoint_episode']
    endpoints = dict(baseline=next(d for d in descriptors if d['sha256'] == S2_SHA and d['arm'] is None))
    for arm in ('control', 'anchor'):
        ends = [d for d in descriptors if d['arm'] == arm and d['episode'] == endpoint]
        if len(ends) == 1:
            endpoints[arm + '500'] = ends[0]
    aliases = {d['id']: d['sha256'] for d in descriptors}
    aliases.update({'control-0': S2_SHA, 'anchor-0': S2_SHA})
    value = dict(format='s2-kl-model-manifest-v1', registration_digest=reg['registration_digest'],
        aliases=aliases, models=list(unique.values()), endpoints=endpoints)
    value['model_manifest_digest'] = canonical_digest(value)
    return value


def candidates_for_run(output, reg):
    from training_state import load_training_state
    output = Path(output).resolve()
    descriptors = [reg['checkpoints']['S2']]
    for arm in ('control', 'anchor'):
        directory = output / 'arms' / arm
        if not (directory / 'completed.json').is_file() or not (directory / 'training_state.pt').is_file():
            raise ValueError('Both training arms must be complete')
        completion = read_json(directory / 'completed.json')
        state = load_training_state(directory / 'training_state.pt', expected_format=FORMAT)
        if (completion != dict(arm=arm, episode=reg['config']['endpoint_episode'],
                registration_digest=reg['registration_digest'], state_sha256=sha(directory / 'training_state.pt')) or
                state['episode'] != reg['config']['endpoint_episode'] or
                state['provenance']['registration_digest'] != reg['registration_digest']):
            raise ValueError('Training completion/state identity drift')
        context = S2KLRun(output, arm, reg)
        context.load_state_dict(state['context'])
        descriptors.extend(state['context']['candidates'])
    manifest = model_manifest(reg, descriptors)
    path = output / 'model_manifest.json'
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError('Model manifest projection drift')
    else:
        atomic_json(path, manifest)
    return manifest['models']
