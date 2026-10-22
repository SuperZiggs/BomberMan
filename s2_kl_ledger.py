"""Atomic offline/board ledgers and reproducible, evidence-backed pilot reports."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import zipfile

from phase_d_s2_ledger import _read_ledger, _rows
from phase_d_s2_protocol import validate_task_payload
from s2_kl_evaluation import (build_report, diagnose_model, load_samples,
    validate_diagnostic, validation_tasks)
from s2_kl_protocol import ROOT, atomic_json, canonical_digest, read_json, sha, verify_run
from s2_kl_training import candidates_for_run
from training_state import run_lock


def process_peak_rss():
    """Peak resident bytes for this process; measured independently for each worker."""
    if os.name != 'nt':
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == 'darwin' else peak * 1024)
    import ctypes
    from ctypes import wintypes
    size = ctypes.c_size_t
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, size) for name in ('peak_rss', 'rss', 'peak_paged', 'paged',
                'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = ctypes.c_void_p
    read = ctypes.windll.psapi.GetProcessMemoryInfo
    read.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD]
    if not read(current(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.peak_rss)


def _models(output, reg, *, required=False):
    complete = all((output / 'arms' / arm / 'completed.json').is_file() for arm in ('control', 'anchor'))
    if complete:
        return candidates_for_run(output, reg)
    if required or (output / 'model_manifest.json').exists():
        raise ValueError('Both training arms must be complete; model manifest cannot be downgraded')
    for arm in ('control', 'anchor'):
        directory = output / 'arms' / arm
        if (directory / 'completed.json').exists() and not (directory / 'training_state.pt').is_file():
            raise ValueError('Completion exists without training state')
    return []


def _diagnostics(output, reg, models):
    expected = {m['sha256']: m for m in models}
    result = {}
    paths = sorted((output / 'diagnostics').glob('*.json'))
    samples = load_samples(reg) if paths else None
    for path in paths:
        if path.stem not in expected or path.stem in result:
            raise ValueError('Unexpected diagnostic shard')
        payload = read_json(path)
        validate_diagnostic(payload, expected[path.stem], reg, samples)
        result[path.stem] = payload
    return result


def _boards(output, reg, tasks):
    saved = _read_ledger(output / 'validation/tasks', tasks)
    roots = {Path(reg['source_root']).resolve(), (output / 'sources').resolve()}
    for payload in saved.values():
        worker = payload.get('worker_runtime', {})
        if (worker.get('device') != 'cpu' or worker.get('threads') != 1 or
                worker.get('deterministic') is not True or
                Path(worker.get('source_root', '')).resolve() not in roots):
            raise ValueError('Worker runtime/source identity drift')
    return saved


def _report(reg, models, diagnostics, saved):
    report = build_report(reg, models, diagnostics, _rows(saved))
    report['evidence'] = dict(
        diagnostics={key: value['diagnostic_digest'] for key, value in sorted(diagnostics.items())},
        boards={key: canonical_digest(value) for key, value in sorted(saved.items())})
    return report


def _check_previous(output, reg, models, diagnostics, saved):
    path = output / 'report.json'
    if not path.exists():
        return
    previous = read_json(path)
    evidence = previous.get('evidence')
    if not isinstance(evidence, dict) or set(evidence) != {'diagnostics', 'boards'}:
        raise ValueError('Report is not backed by raw evidence')
    old_diagnostics, old_saved = {}, {}
    for key, digest in evidence['diagnostics'].items():
        if key not in diagnostics or diagnostics[key]['diagnostic_digest'] != digest:
            raise ValueError('Prior diagnostic evidence drift')
        old_diagnostics[key] = diagnostics[key]
    for key, digest in evidence['boards'].items():
        if key not in saved or canonical_digest(saved[key]) != digest:
            raise ValueError('Prior board evidence drift')
        old_saved[key] = saved[key]
    old_models = models if previous.get('model_manifest_digest') is not None else []
    if _report(reg, old_models, old_diagnostics, old_saved) != previous:
        raise ValueError('Report projection/raw mismatch')


def _decision_text(report):
    text = '# S2-KL decision\n\nStatus: ' + report['status'] + '\n\n'
    text += f"Complete: {report['complete']}; RB games: {report['counts']['games']}/{report['counts']['expected_games']}\n\n"
    text += 'S2 remains the default. This pilot has no held-out stage or Phase D qualification.\n\n'
    text += '| Model | R | Q | WR | ActiveWR | Draw | Suicide | SD win | SD death | Passive |\n'
    text += '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for key, entry in report['summary'].items():
        offline, rb = entry.get('offline', {}), entry.get('RB', {})
        values = [offline.get('R'), offline.get('Q'), *[rb.get(k + '_rate') for k in
            ('win', 'active_win', 'draw', 'suicide', 'sudden_death_win', 'sudden_death_death', 'passive_win')]]
        text += '| ' + key + ' | ' + ' | '.join('—' if v is None else f'{v:.4f}' for v in values) + ' |\n'
    text += '\nConditions: ' + str(report['conditions']) + '\n'
    text += '\n' + '\n'.join('- ' + item for item in report['limitations']) + '\n'
    return text


def _write_report(output, reg, models, diagnostics, saved):
    _check_previous(output, reg, models, diagnostics, saved)
    report = _report(reg, models, diagnostics, saved)
    path = output / 'report.json'
    if not path.exists() or read_json(path) != report:
        atomic_json(path, report, overwrite=True)
    text = _decision_text(report)
    decision = output / 'decision_report.md'
    if not decision.exists() or decision.read_text(encoding='utf-8') != text:
        temporary = decision.with_suffix('.md.tmp')
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, decision)
    return report


def _frozen_job(job):
    from phase_d_s2_evaluation import evaluate_board
    output, task = job
    output = Path(output)
    verify_run(output)
    payload = evaluate_board(task)
    payload['worker_runtime'].update(pid=os.getpid(), peak_rss_bytes=process_peak_rss())
    validate_task_payload(task, payload)
    atomic_json(output / 'validation/tasks' / (task['id'] + '.json'), payload)
    return task['id']


def diagnose_run(output, *, resume=False):
    from progress import bar
    output = Path(output).resolve()
    with run_lock(output):
        reg = verify_run(output)
        models = _models(output, reg, required=True)
        diagnostics = _diagnostics(output, reg, models)
        tasks = validation_tasks(reg, models)
        saved = _boards(output, reg, tasks)
        _check_previous(output, reg, models, diagnostics, saved)
        if diagnostics and not resume:
            raise FileExistsError('Diagnostics exist; use --resume')
        with bar(len(models), 'S2-KL offline', len(diagnostics)) as shown:
            try:
                for model in models:
                    if model['sha256'] in diagnostics:
                        continue
                    verify_run(output)
                    payload = diagnose_model(model, reg)
                    validate_diagnostic(payload, model, reg, load_samples(reg))
                    atomic_json(output / 'diagnostics' / (model['sha256'] + '.json'), payload)
                    diagnostics[model['sha256']] = payload
                    shown.update(1)
            except BaseException:
                diagnostics = _diagnostics(output, reg, models)
                _write_report(output, reg, models, diagnostics, saved)
                raise
        report = _write_report(output, reg, models, diagnostics, saved)
        return dict(status='DIAGNOSTICS_COMPLETE', models=len(diagnostics), report_status=report['status'])


def validate_run(output, *, resume=False):
    from progress import bar
    output = Path(output).resolve()
    with run_lock(output):
        reg = verify_run(output)
        models = _models(output, reg, required=True)
        diagnostics = _diagnostics(output, reg, models)
        if set(diagnostics) != {model['sha256'] for model in models}:
            raise ValueError('Complete diagnostics required before validation')
        tasks = validation_tasks(reg, models)
        saved = _boards(output, reg, tasks)
        _check_previous(output, reg, models, diagnostics, saved)
        if saved and not resume:
            raise FileExistsError('Validation ledger exists; use --resume')
        missing = [t for t in tasks if t['id'] not in saved]
        with bar(2 * len(tasks), 'S2-KL validation', 2 * len(saved)) as shown:
            try:
                if reg['config']['workers'] == 1:
                    for task in missing:
                        _frozen_job((str(output), task))
                        shown.update(2)
                elif missing:
                    import multiprocessing
                    with multiprocessing.get_context('spawn').Pool(reg['config']['workers']) as pool:
                        for _ in pool.imap_unordered(_frozen_job, [(str(output), t) for t in missing], chunksize=1):
                            shown.update(2)
            except BaseException:
                saved = _boards(output, reg, tasks)
                _write_report(output, reg, models, diagnostics, saved)
                raise
        saved = _boards(output, reg, tasks)
        report = _write_report(output, reg, models, diagnostics, saved)
        projection = dict(registration_digest=reg['registration_digest'],
            model_manifest_digest=report['model_manifest_digest'], counts=report['counts'], summary=report['summary'],
            ledger_digest=canonical_digest(saved))
        path = output / 'validation.json'
        if path.exists() and read_json(path) != projection:
            raise ValueError('Validation projection/raw mismatch')
        if not path.exists():
            atomic_json(path, projection)
        return report


def package_run(output):
    output = Path(output).resolve()
    with run_lock(output):
        reg = verify_run(output)
        models = _models(output, reg)
        diagnostics = _diagnostics(output, reg, models)
        tasks = validation_tasks(reg, models)
        saved = _boards(output, reg, tasks)
        report = _write_report(output, reg, models, diagnostics, saved)
        validation = output / 'validation.json'
        if validation.exists():
            projection = dict(registration_digest=reg['registration_digest'],
                model_manifest_digest=report['model_manifest_digest'], counts=report['counts'], summary=report['summary'],
                ledger_digest=canonical_digest(saved))
            if read_json(validation) != projection:
                raise ValueError('Validation projection/raw mismatch')
        destination = output.with_name(output.name + '-send.zip')
        temporary = destination.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_symlink() or getattr(path, 'is_junction', lambda: False)() or not path.resolve().is_relative_to(output):
                    raise ValueError('Symlink/junction outside package closure')
                if not path.is_file() or path.suffix in ('.tmp', '.lock', '.pyc') or '__pycache__' in path.parts:
                    continue
                if path.suffix == '.npz':
                    raise ValueError('Collection shards are not pilot output')
                archive.write(path, str(Path(output.name) / path.relative_to(output)))
        with temporary.open('r+b') as handle:
            handle.flush()
            os.fsync(handle.fileno())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('Package CRC validation failed')
        os.replace(temporary, destination)
        return report
