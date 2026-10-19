"""Atomic Phase C training snapshots; inference checkpoints remain plain state_dicts."""
import hashlib
from contextlib import contextmanager
import os
from pathlib import Path
import random

import numpy as np
import torch


@contextmanager
def run_lock(output):
    """OS-released lock: an interrupted process cannot leave a stale writer lock."""
    path = Path(str(output) + '-run.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            lock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as error:
            raise ValueError('Run is already active; refusing a second writer/resume') from error
        try:
            yield
        finally:
            handle.seek(0)
            unlock()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_training_state(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or state.get('format') != 'bomberman_phase_c_training_v1':
        raise ValueError('Expected a full Phase C training state, not an inference checkpoint')
    if not isinstance(state.get('episode'), int) or state['episode'] < 0:
        raise ValueError('Invalid completed episode in training state')
    if not {'provenance', 'model', 'optimizer', 'scheduler', 'histories', 'selection', 'rng',
            'training_mode', 'context'} <= state.keys():
        raise ValueError('Incomplete training state')
    if state['episode'] > state['provenance']['config']['episodes']:
        raise ValueError('Completed episode exceeds the registered schedule')
    return state


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        'training_state.py', 'train_actor_critic.py', 'bomberman_env.py', 'rule_based_bot.py',
        'tactical_teacher.py', 'combat_move_teacher.py', 'evaluate.py', 'kill_search.py',
        'group_rl.py')}


def runtime_state():
    return {'torch': str(torch.__version__), 'numpy': np.__version__, 'threads': torch.get_num_threads(),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def capture_rng(env):
    np_state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (np_state[0], torch.tensor(np_state[1].astype(np.int64)),
            *np_state[2:]), 'env': env.rng.bit_generator.state, 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, env):
    random.setstate(state['python'])
    np_state = state['numpy']
    np.random.set_state((np_state[0], np_state[1].numpy().astype(np.uint32), *np_state[2:]))
    env.rng.bit_generator.state = state['env']
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from the saved RNG state')
        torch.cuda.set_rng_state_all(state['cuda'])
