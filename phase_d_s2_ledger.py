"""Atomic, verified board ledgers and explicit D-S2 evaluation stages."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
import zipfile

from phase_d_s2_protocol import (atomic_json, canonical_digest, read_json, sha,
    validate_task_payload, verify_run)
from phase_d_s2_evaluation import (build_report, build_test_manifest, candidates_for_run,
    evaluate_board, select_models, validation_tasks)
from training_state import run_lock


def _read_ledger(directory, tasks):
    expected={t['id']:t for t in tasks}
    if len(expected)!=len(tasks):raise ValueError('Duplicate expected task identity')
    saved={}
    for path in sorted(directory.glob('*.json')):
        payload=read_json(path);key=payload.get('id')
        if key not in expected or key in saved:raise ValueError('Unexpected or duplicate ledger task')
        if path.name!=key+'.json':raise ValueError('Ledger filename/identity mismatch')
        validate_task_payload(expected[key],payload);saved[key]=payload
    return saved


def _frozen_job(job):
    """Imported from the registered sources tree by the frozen parent CLI."""
    task,destination=job
    payload=evaluate_board(task)
    atomic_json(Path(destination)/(task['id']+'.json'),payload)
    return task['id']


def run_tasks(output,stage,tasks,workers,*,resume=False,evaluator=None,commit_hook=None):
    from progress import bar
    directory=Path(output)/stage/'tasks'
    saved=_read_ledger(directory,tasks)
    missing=[t for t in tasks if t['id'] not in saved]
    if saved and not resume:raise ValueError('Ledger already exists; use --resume')
    directory.mkdir(parents=True,exist_ok=True)
    # Injected evaluators are deliberately serial: production callers cannot select one in the CLI.
    if evaluator is not None or workers==1:
        with bar(2*len(tasks),'D-S2 '+stage,2*len(saved)) as shown:
            for task in missing:
                payload=(evaluator or evaluate_board)(deepcopy(task))
                validate_task_payload(task,payload)
                if commit_hook:commit_hook('before_task_commit',task['id'])
                atomic_json(directory/(task['id']+'.json'),payload)
                if commit_hook:commit_hook('after_task_commit',task['id'])
                shown.update(2)
    elif missing:
        # Pool's public context manager terminates unfinished workers on Ctrl+C/error.
        # progress.parallel_map waits for every submitted future on interruption.
        import multiprocessing
        jobs=[(t,str(directory)) for t in missing]
        with bar(2*len(tasks),'D-S2 '+stage,2*len(saved)) as shown:
            with multiprocessing.get_context('spawn').Pool(workers) as pool:
                for _ in pool.imap_unordered(_frozen_job,jobs,chunksize=1):shown.update(2)
    return _read_ledger(directory,tasks)


def _rows(saved):
    return [row for key in sorted(saved) for row in saved[key]['records']]


def _incomplete(stage,actual,expected,status='INCOMPLETE'):
    return dict(stage=stage,status=status,complete=False,actual_games=actual,expected_games=expected,
        summary={},comparisons={},pilot_gate={'pass':False,'status':'NOT_EVALUATED'},
        phase_d_gate={'pass':False,'status':'NOT_EVALUATED'},limitations=['Evaluation is incomplete.'])


def _decision(output,report):
    text='# D-S2 decision\n\nStatus: '+report['status']+'\n\n'
    text+='Complete: '+str(report['complete'])+'; games: '+str(report['actual_games'])+'/'+str(report['expected_games'])+'\n\n'
    text+='Default checkpoint remains S2 until a full experiment meets its registered gates.\n'
    for name in ('pilot_gate','phase_d_gate'):
        text+='\n'+name+': '+str(report[name])+'\n'
    path=Path(output)/'decision_report.md';temporary=path.with_suffix('.md.tmp')
    if path.exists() and path.read_text(encoding='utf-8')==text:return
    with temporary.open('w',encoding='utf-8') as handle:
        handle.write(text);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def _selection(output,reg,candidates,snapshots,tasks):
    saved=_read_ledger(Path(output)/'validation/tasks',tasks)
    if len(saved)!=len(tasks):raise ValueError('Incomplete validation ledger')
    selection=select_models(_rows(saved),candidates)
    selection.update(mode=reg['mode'],registration_digest=reg['registration_digest'],
                     validation_ledger_digest=canonical_digest(saved))
    selection['test_manifest']=build_test_manifest(reg,selection['models'],snapshots) if selection['open_test'] else []
    selection['selection_digest']=canonical_digest(selection)
    # Selection digest is outside task identity, avoiding a self-referential JSON digest.
    return selection


def evaluate_stage(output,stage,*,resume=False,evaluator=None,commit_hook=None):
    if stage not in ('validation','test'):raise ValueError('Unknown evaluation stage')
    output=Path(output).resolve()
    with run_lock(output):
        reg=verify_run(output);candidates,snapshots=candidates_for_run(output,reg)
        val_tasks=validation_tasks(output)
        selected_path=output/'selected.json';opened_path=output/'test_opened.json'
        if stage=='validation':
            if selected_path.exists() and not resume:raise ValueError('Validation selected; use --resume')
            try:
                saved=run_tasks(output,stage,val_tasks,reg['config']['workers'],resume=resume,
                    evaluator=evaluator,commit_hook=commit_hook)
            except BaseException:
                saved=_read_ledger(output/stage/'tasks',val_tasks)
                atomic_json(output/'report.json',_incomplete(stage,2*len(saved),2*len(val_tasks)),overwrite=True)
                raise
            selected=_selection(output,reg,candidates,snapshots,val_tasks)
            if selected_path.exists():
                if read_json(selected_path)!=selected:raise ValueError('Selection/validation ledger drift')
            else:atomic_json(selected_path,selected)
            validation=dict(models=selected['models'],
                validation_summary=selected['validation_summary'],actual_games=2*len(saved),
                expected_games=2*len(val_tasks),registration_digest=reg['registration_digest'])
            if not (output/'validation.json').exists():atomic_json(output/'validation.json',validation)
            elif read_json(output/'validation.json')!=validation:raise ValueError('Validation projection drift')
            if opened_path.exists():return selected
            report=_incomplete(stage,2*len(saved),2*len(val_tasks),
                'TEST_READY' if selected['open_test'] else 'VALIDATION_STOP')
            report.update(complete=True,summary=selected['validation_summary'],
                          limitations=['Validation only; held-out test has not run.'])
            if not (output/'report.json').exists() or read_json(output/'report.json')!=report:
                atomic_json(output/'report.json',report,overwrite=True)
            _decision(output,report)
            return selected
        # Recompute from validated raw rows, rather than trusting a mutable selection file.
        selected=_selection(output,reg,candidates,snapshots,val_tasks)
        if not selected_path.is_file() or read_json(selected_path)!=selected:raise ValueError('Selection drift')
        if not selected['open_test']:raise ValueError('VALIDATION_STOP: test is closed')
        tasks=deepcopy(selected['test_manifest'])
        for task in tasks:task['selection_digest']=selected['selection_digest']
        # Check every required weight before the one-open marker, including old snapshots.
        for task in tasks:
            for desc in (task['learner'],task['opponent']):
                if desc['checkpoint'] is not None and sha(desc['checkpoint'])!=desc['sha256']:
                    raise ValueError('Test checkpoint drift')
        marker=dict(format='d-s2-test-opened-v1',registration_digest=reg['registration_digest'],
            selection_digest=selected['selection_digest'],models=selected['models'],task_manifest=tasks)
        _read_ledger(output/'test/tasks',tasks)
        if opened_path.exists():
            if not resume:raise ValueError('Test already opened; use --resume')
            if read_json(opened_path)!=marker:raise ValueError('Test-open marker drift')
        else:
            if resume:raise ValueError('Test has never opened; omit --resume')
            if (output/'test/tasks').exists():raise ValueError('Test ledger without open marker')
            atomic_json(opened_path,marker)
        try:
            saved=run_tasks(output,'test',tasks,reg['config']['workers'],resume=resume,
                evaluator=evaluator,commit_hook=commit_hook)
        except BaseException:
            saved=_read_ledger(output/'test/tasks',tasks)
            atomic_json(output/'report.json',_incomplete('test',2*len(saved),2*len(tasks)),overwrite=True)
            raise
        report=build_report(_rows(saved),selected)
        report.update(registration_digest=reg['registration_digest'],selection_digest=selected['selection_digest'])
        # Completed resume compares byte-stable content and leaves the existing artifact untouched.
        report_path=output/'report.json'
        if not report_path.exists() or read_json(report_path)!=report:atomic_json(report_path,report,overwrite=True)
        _decision(output,report)
        return report


def package_run(output):
    output=Path(output).resolve()
    with run_lock(output):
        reg=verify_run(output)
        if (output/'selected.json').exists():
            candidates,snapshots=candidates_for_run(output,reg)
            selected=_selection(output,reg,candidates,snapshots,validation_tasks(output))
            if read_json(output/'selected.json')!=selected:raise ValueError('Selection drift')
            if (output/'test_opened.json').exists():
                tasks=deepcopy(selected['test_manifest'])
                for task in tasks:task['selection_digest']=selected['selection_digest']
                marker=read_json(output/'test_opened.json')
                expected=dict(format='d-s2-test-opened-v1',registration_digest=reg['registration_digest'],
                    selection_digest=selected['selection_digest'],models=selected['models'],task_manifest=tasks)
                if marker!=expected:raise ValueError('Test-open marker drift')
                saved=_read_ledger(output/'test/tasks',tasks)
                if (output/'report.json').exists() and read_json(output/'report.json').get('complete'):
                    if len(saved)!=len(tasks):raise ValueError('Incomplete test ledger with completed report')
                    expected=build_report(_rows(saved),selected)
                    expected.update(registration_digest=reg['registration_digest'],selection_digest=selected['selection_digest'])
                    if read_json(output/'report.json')!=expected:raise ValueError('Report drift')
        if not (output/'report.json').exists():
            atomic_json(output/'report.json',_incomplete('registered',0,0))
        report=read_json(output/'report.json')
        _decision(output,report)
        destination=output.with_name(output.name+'-send.zip');temporary=destination.with_suffix('.zip.tmp')
        try:
            with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(output.rglob('*')):
                    if path.is_symlink():raise ValueError('Symlink outside package closure')
                    if path.is_file() and path.suffix not in ('.tmp','.lock','.pyc') and '__pycache__' not in path.parts:
                        if not path.resolve().is_relative_to(output):raise ValueError('Package path escapes output')
                        archive.write(path,path.relative_to(output).as_posix())
            os.replace(temporary,destination)
        finally:
            if temporary.exists():temporary.unlink()
        return dict(archive=str(destination),status=report['status'],registration_digest=reg['registration_digest'])
