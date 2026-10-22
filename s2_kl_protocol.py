"""Immutable inputs, seed reservations and source snapshots for the S2-KL pilot."""
from __future__ import annotations

import ast
from contextlib import nullcontext
import os
from pathlib import Path
import shutil
import subprocess

from phase_d_s2_protocol import (atomic_json, canonical_digest, configure_runtime,
    inventory_seeds, read_json, runtime, sha)

ROOT = Path(__file__).resolve().parent
SPEC = 'docs/superpowers/specs/2026-10-23-s2-kl-retention-design.md'
S2_REL = 'runs/sd8a-20261020/train/epoch_16.pt'
S2_SHA = 'd222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814'
SAMPLE_REL = 'runs/phaseD-s2-20261022-analysis-20261023/offline_sample_manifest.json'
SAMPLE_SHA = '9b36b8e07ed8f65175e5c9fbf53a18eca16ae791f0073d19730b28e75529e333'
REGISTRY = 'runs/phase_d_s2_seed_registry.json'


def config_for(smoke=False):
    train = dict(phase='D', episodes=500, board_size=9, num_agents=2, max_steps=500,
        gamma=.99, lr=2.5e-5, value_coef=.5, entropy_coef=.0075, wall_destroy_bonus=.15,
        snapshot_every=100, seed=2026102301, validation_every=0, validation_episodes=2,
        attack_teacher='local', teacher_version='v2', normalize_advantage=False,
        advantage_estimator='mc', policy_objective='ac', sudden_death_probability=1.,
        sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40,
        attack_lookahead=2, min_pressure_gain=.15, trap_sequence_depth=2,
        sequence_pressure_gain=.35, escape_loss_coef=.5, combat_loss_coef=.9,
        navigation_loss_coef=.4, ring_safety_coef=.6, late_game_start=350,
        late_game_attack_multiplier=1.75, search_teacher=False, group_rl=False,
        advantage_diagnostics=True, reward_diagnostics=False, run_report=False, log_every=50)
    if smoke:
        train.update(episodes=4, max_steps=20, snapshot_every=2, log_every=4)
    return dict(train=train, anchor_coef={'control': 0., 'anchor': 1.},
        checkpoint_episodes=[0, 2, 4] if smoke else [0, 100, 250, 500],
        endpoint_episode=4 if smoke else 500, workers=2)


def seed_manifest(smoke=False):
    spans = ((790000, 790001), (790002, 790003)) if smoke else (
        (710000, 710249), (720000, 720099))
    return dict(mode='smoke' if smoke else 'full', ranges={
        key: dict(start=a, end_inclusive=b) for key, (a, b) in zip(('train', 'validation'), spans)})


def input_path(root, relative):
    root = Path(root).resolve()
    path = root / str(relative).replace('\\', '/')
    if not path.resolve().is_relative_to(root):
        raise ValueError('Input path outside registered root')
    return path


def source_closure():
    # The console helper is a runtime dependency of the staged train/worker entrypoints.
    pending, found = ['run_s2_kl_probe', 'progress'], {}
    while pending:
        name = pending.pop()
        path = ROOT / (name + '.py')
        if path.name in found or not path.is_file():
            continue
        found[path.name] = sha(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = ([v.name for v in node.names] if isinstance(node, ast.Import) else
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str) and
                    ((isinstance(node.func, ast.Name) and node.func.id == '__import__') or
                     (isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module'))):
                names.append(node.args[0].value)
            pending.extend(n.split('.')[0] for n in names if (ROOT / (n.split('.')[0] + '.py')).is_file())
    return dict(sorted(found.items()))


def _inputs(input_root):
    from phase_d_s2_evaluation import load_registered_model
    root = Path(input_root)
    checkpoint = input_path(root, S2_REL)
    if sha(checkpoint) != S2_SHA:
        raise ValueError('S2 checkpoint hash mismatch')
    selected = read_json(root / 'runs/sd8a-20261020/selected.json')
    opened = read_json(root / 'runs/sd8a-20261020/test_opened.json')
    if (str(selected.get('checkpoint', '')).replace('\\', '/') != S2_REL or
            opened.get('models', {}).get('S2') != S2_SHA):
        raise ValueError('S2 selected/test-opened identity mismatch')
    load_registered_model(checkpoint, 'S2')  # strict 13-channel / 9x9 tensor contract
    manifest_path = input_path(root, SAMPLE_REL)
    if sha(manifest_path) != SAMPLE_SHA:
        raise ValueError('Locked sample manifest hash mismatch')
    manifest = read_json(manifest_path)
    for relative, expected in manifest['input_sha256'].items():
        if sha(input_path(root, relative)) != expected:
            raise ValueError('Collection shard hash mismatch: ' + relative)
    metadata = {name: sha(root / 'runs/sd8a-20261020' / name) for name in
        ('selected.json', 'test_opened.json', 'registration.json', 'collection_summary.json')}
    return dict(S2=dict(id='S2', arm=None, episode=0, checkpoint=str(checkpoint.resolve()), sha256=S2_SHA)), dict(
        relative_path=SAMPLE_REL, sha256=SAMPLE_SHA, shards=manifest['input_sha256'], metadata=metadata)


def _inventory(input_root, exclude):
    """Include raw task filenames and nested UI metadata skipped by the legacy scanner."""
    roots = [Path(input_root), Path(input_root) / 'game UI', ROOT / 'game UI']
    roots = list(dict.fromkeys(p.resolve() for p in roots))
    result = dict(ranges=[], unresolved=[], sources={}, rng_seeds=[])
    for root in roots:
        old = inventory_seeds(root, exclude=exclude)
        prefix = str(root)
        result['ranges'].extend([a, b, prefix + '/' + s] for a, b, s in old['ranges'])
        result['unresolved'].extend(prefix + '/' + s for s in old['unresolved'])
        result['sources'].update({prefix + '/' + k: v for k, v in old['sources'].items()})
        result['rng_seeds'].extend(old['rng_seeds'])

        def visit(value, source):
            if isinstance(value, dict):
                # Historical collection/export indexes declare inclusive endpoints.
                for first, last in (('first_seed', 'last_seed'),
                                    ('collect_first_seed', 'collect_last_seed'), ('seed_min', 'seed_max')):
                    if first in value and last in value:
                        a, b = value[first], value[last]
                        if type(a) is int and type(b) is int and 0 <= a <= b:
                            result['ranges'].append([a, b, source])
                        else:
                            result['unresolved'].append(source + ': invalid seed interval')
                for key, item in value.items():
                    if key in ('board', 'board_seed', 'normal_seed', 'validation_seed', 'test_seed',
                               'train_seed', 'mechanism_seed', 'collect_seed', 'seed', 'evaluation_seed',
                               'selection_seed', 'first_seed', 'last_seed', 'last_board_seed',
                               'collect_first_seed', 'collect_last_seed', 'seed_min', 'seed_max',
                               'bootstrap_seed', 'original_heldout_seed'):
                        rng = (key in ('bootstrap_seed', 'original_heldout_seed') or
                               key == 'seed' and ('phase' in value and 'episodes' in value or
                                                  value.get('train_only') is True and 'samples' in value))
                        if rng:
                            if type(item) is not int:
                                result['unresolved'].append(source + ': invalid training RNG')
                            else:
                                result['rng_seeds'].append(dict(source=source, seed=item, kind='rng'))
                        elif type(item) is int and item >= 0:
                            # A scalar board/seed is one board, never its own range length.
                            count = (value.get(key[:-5] + '_games', value.get(key[:-5] + '_episodes', 1))
                                     if key.endswith('_seed') else 1)
                            if type(count) is not int or count <= 0:
                                result['unresolved'].append(source + ': invalid seed count ' + key)
                            else:
                                result['ranges'].append([item, item + count - 1, source])
                        else:
                            result['unresolved'].append(source + ': invalid seed ' + key)
                    elif key in ('board_ids', 'board_seeds') or key.endswith(('_board_ids', '_board_seeds')):
                        if isinstance(item, list) and all(type(n) is int and n >= 0 for n in item):
                            result['ranges'].extend([n, n, source] for n in item)
                        else:
                            result['unresolved'].append(source + ': invalid board IDs')
                    elif key in ('seeds', 'rollout_seeds', 'skill_seeds'):
                        if isinstance(item, list) and item and all(type(n) is int and n >= 0 for n in item):
                            # Old evaluation exports use seeds=[first,last], conservatively inclusive.
                            if key == 'seeds':
                                result['ranges'].append([min(item), max(item), source])
                            else:
                                result['rng_seeds'].extend(dict(source=source, seed=n, kind='scenario_rng') for n in item)
                        else:
                            result['unresolved'].append(source + ': invalid seeds')
                    elif key.endswith(('_seed', '_seeds')) or key == 'seed_scheme':
                        result['unresolved'].append(source + ': unknown seed mapping ' + key)
                    visit(item, source)
            elif isinstance(value, list):
                for item in value:
                    visit(item, source)

        for path in sorted([*(root / 'runs').rglob('*.json'), *(root / 'runs').rglob('*.jsonl')]):
            if path.name == Path(REGISTRY).name or any(path.resolve().is_relative_to(Path(p).resolve()) for p in exclude):
                continue
            source = str(path)
            try:
                if path.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError('metadata requires verified seed index')
                if path.suffix == '.jsonl':
                    with path.open(encoding='utf-8') as handle:
                        for line in handle:
                            if line.strip():
                                import json
                                visit(json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s))), source)
                else:
                    visit(read_json(path), source)
                result['sources'][source] = sha(path)
            except (OSError, ValueError) as error:
                result['unresolved'].append(source + ': ' + str(error))
    return result


def _check_history(manifest, inventory, book, experiment_id):
    if inventory['unresolved']:
        raise ValueError('Unresolved seed history: ' + str(inventory['unresolved'][:8]))
    for span in manifest['ranges'].values():
        if any(span['start'] <= b and a <= span['end_inclusive'] for a, b, _ in inventory['ranges']):
            raise ValueError('Seed overlap with historical boards')
        for name, entry in book.get('reservations', {}).items():
            if name == experiment_id:
                continue
            if any(span['start'] <= old['end_inclusive'] and old['start'] <= span['end_inclusive']
                   for old in entry['ranges'].values()):
                raise ValueError('Seed overlap with existing reservation')


def register(output, input_root, *, device='cuda:0', workers=2, smoke=False, resume=False):
    from training_state import run_lock
    output, input_root = Path(output).resolve(), Path(input_root).resolve()
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError('workers must be 1 or 2')
    configure_runtime(device)
    registry = input_root / REGISTRY
    pending = output.with_name(output.name + '.registration-pending')
    with run_lock(output):
        with (run_lock(registry) if not smoke else nullcontext()):
            if output.exists():
                if not resume:
                    raise FileExistsError('Output exists; use --resume')
                reg = verify_run(output, allow_pending=True)
                if (reg['input_root'], reg['device'], reg['mode'], reg['config']['workers']) != (
                        str(input_root), device, 'smoke' if smoke else 'full', workers):
                    raise ValueError('Registration arguments drift')
                if not smoke:
                    book = read_json(registry)
                    book['reservations'][reg['experiment_id']]['status'] = 'COMPLETE'
                    atomic_json(registry, book, overwrite=True)
                return verify_run(output)
            checkpoints, offline = _inputs(input_root)
            manifest = seed_manifest(smoke)
            config = config_for(smoke)
            config['workers'] = workers
            experiment_id = 'S2-KL:' + str(output)
            book = read_json(registry) if registry.exists() else {'reservations': {}}
            inventory = dict(ranges=[], unresolved=[], sources={}, rng_seeds=[]) if smoke else _inventory(input_root, (output, pending))
            if not smoke:
                _check_history(manifest, inventory, book, experiment_id)
            manifest['inventory_digest'] = canonical_digest(inventory)
            reg = dict(format='s2-kl-registration-v1', experiment_id=experiment_id, mode=manifest['mode'],
                output=str(output), input_root=str(input_root), source_root=str(ROOT), device=device,
                config=config, checkpoints=checkpoints, offline_manifest=offline, sources=source_closure(),
                spec_sha256=sha(ROOT / SPEC), runtime=runtime(), seed_manifest=manifest,
                git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
            reg['registration_digest'] = canonical_digest(reg)
            if pending.exists():
                if not resume:
                    raise FileExistsError('Pending registration exists; use --resume')
                identity = pending / 'registration.json'
                if identity.exists():
                    old = read_json(identity)
                    if old != reg:
                        raise ValueError('Pending registration identity drift')
                    reg = old
                elif experiment_id in book.get('reservations', {}):
                    raise ValueError('Reserved registration lost identity')
            else:
                pending.mkdir(parents=True)
            if not (pending / 'registration.json').exists():
                atomic_json(pending / 'registration.json', reg)
            if not smoke:
                own = book['reservations'].get(experiment_id)
                if own and (not resume or own.get('registration_digest') != reg['registration_digest']):
                    raise ValueError('Conflicting reservation identity')
                book['reservations'][experiment_id] = dict(status='PENDING', ranges=manifest['ranges'],
                    registration_digest=reg['registration_digest'], output=str(output))
                atomic_json(registry, book, overwrite=True)
            for name, expected in {**reg['sources'], 'spec.md': reg['spec_sha256']}.items():
                source = ROOT / SPEC if name == 'spec.md' else ROOT / name
                target = pending / 'sources' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                copying = target.with_suffix(target.suffix + '.copy.tmp')
                shutil.copy2(source, copying)
                if sha(copying) != expected:
                    raise ValueError('Source changed during registration: ' + name)
                with copying.open('r+b') as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(copying, target)
            shutil.copy2(input_root / SAMPLE_REL, pending / 'offline_sample_manifest.json')
            for name, value in [('runtime.json', reg['runtime']), ('seed_manifest.json', manifest),
                                ('seed_inventory.json', inventory)]:
                path = pending / name
                if path.exists() and read_json(path) != value:
                    raise ValueError('Prepared projection drift: ' + name)
                if not path.exists():
                    atomic_json(path, value)
            (pending / 'logs').mkdir(exist_ok=True)
            verify_run(pending, allow_pending=True)
            os.replace(pending, output)
            if not smoke:
                book['reservations'][experiment_id]['status'] = 'COMPLETE'
                atomic_json(registry, book, overwrite=True)
            return verify_run(output)


def verify_run(output, *, allow_pending=False):
    output = Path(output).resolve()
    reg = read_json(output / 'registration.json')
    if reg.get('format') != 's2-kl-registration-v1' or canonical_digest(reg, 'registration_digest') != reg.get('registration_digest'):
        raise ValueError('Registration digest drift')
    registered_output = Path(reg['output']).resolve()
    allowed = {registered_output}
    if allow_pending:
        allowed.add(registered_output.with_name(registered_output.name + '.registration-pending'))
    if (output not in allowed or reg['output'] != str(registered_output) or
            reg['experiment_id'] != 'S2-KL:' + str(registered_output)):
        raise ValueError('Registered output/experiment identity mismatch')
    configure_runtime(reg['device'])
    if runtime() != reg['runtime']:
        raise ValueError('Runtime drift')
    expected_config = config_for(reg['mode'] == 'smoke')
    expected_config['workers'] = reg['config']['workers']
    if (reg['mode'] not in ('smoke', 'full') or type(reg['config']['workers']) is not int or
            reg['config']['workers'] not in (1, 2) or expected_config != reg['config'] or
            reg['seed_manifest']['ranges'] != seed_manifest(reg['mode'] == 'smoke')['ranges']):
        raise ValueError('Config drift')
    checkpoints, offline = _inputs(reg['input_root'])
    if checkpoints != reg['checkpoints'] or offline != reg['offline_manifest']:
        raise ValueError('Input identity drift')
    if sha(output / 'offline_sample_manifest.json') != SAMPLE_SHA:
        raise ValueError('Copied sample manifest drift')
    if (read_json(output / 'runtime.json') != reg['runtime'] or
            read_json(output / 'seed_manifest.json') != reg['seed_manifest'] or
            canonical_digest(read_json(output / 'seed_inventory.json')) != reg['seed_manifest']['inventory_digest']):
        raise ValueError('Runtime/seed projection drift')
    source_root = Path(reg['source_root'])
    if sha(source_root / SPEC) != reg['spec_sha256'] or sha(output / 'sources/spec.md') != reg['spec_sha256']:
        raise ValueError('Spec drift')
    for name, expected in reg['sources'].items():
        if any(sha(path / name) != expected for path in (source_root, output / 'sources', ROOT)):
            raise ValueError('Source drift: ' + name)
    if reg['mode'] == 'full':
        book = read_json(Path(reg['input_root']) / REGISTRY)
        entry = book['reservations'].get(reg['experiment_id'], {})
        if (entry.get('status') not in (('PENDING', 'COMPLETE') if allow_pending else ('COMPLETE',)) or
                entry.get('registration_digest') != reg['registration_digest'] or
                entry.get('ranges') != reg['seed_manifest']['ranges'] or entry.get('output') != reg['output']):
            raise ValueError('Incomplete or drifted seed reservation')
    return reg
