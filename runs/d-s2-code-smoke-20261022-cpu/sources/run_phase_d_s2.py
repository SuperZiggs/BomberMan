"""Controlled D-S2 stages. Full training requires an explicitly registered run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
from phase_d_s2_protocol import (ROOT, config_for, canonical_digest, register, verify_run,
    validate_task_payload)
from phase_d_s2_evaluation import (select_candidates, build_test_manifest,
    paired_macro_comparison, decide_gates, load_registered_model)
from phase_d_s2_ledger import evaluate_stage, package_run


def build_train_schedule(*args,**kwargs):
    from phase_d_s2_training import build_train_schedule as run
    return run(*args,**kwargs)


def train_arm(*args,**kwargs):
    from phase_d_s2_training import train_arm as run
    return run(*args,**kwargs)


def read_training_state(*args,**kwargs):
    from phase_d_s2_training import read_training_state as run
    return run(*args,**kwargs)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    for stage in ('register','train','validate','test','package'):
        p=sub.add_parser(stage);p.add_argument('--output',type=Path,required=True)
        if stage!='package':p.add_argument('--resume',action='store_true')
        if stage=='register':
            p.add_argument('--input-root',type=Path,default=ROOT)
            p.add_argument('--device',choices=('cpu','cuda','cuda:0'),default='cuda:0')
            p.add_argument('--workers',type=int,choices=range(1,6),default=2)
            p.add_argument('--smoke',action='store_true')
        if stage=='train':p.add_argument('--arm',choices=('control','pool'),required=True)
    args=parser.parse_args(argv)
    try:
        if args.stage!='register':
            # Verify live bytes first; all stage imports and spawned workers then use this frozen copy.
            reg=verify_run(args.output)
            frozen=args.output.resolve()/'sources/run_phase_d_s2.py'
            if Path(__file__).resolve()!=frozen:
                import subprocess,sys
                return subprocess.call([sys.executable,str(frozen),*(argv if argv is not None else sys.argv[1:])])
        if args.stage=='register':result=register(args.output,args.input_root,args.device,args.workers,args.smoke,args.resume)
        elif args.stage=='train':result=train_arm(args.output,args.arm,resume=args.resume)
        elif args.stage in ('validate','test'):result=evaluate_stage(args.output,'validation' if args.stage=='validate' else 'test',resume=args.resume)
        else:result=package_run(args.output)
        print(json.dumps({'stage':args.stage,'output':str(args.output),'status':(result or {}).get('status','OK')},default=str))
        return 0
    except (ValueError,OSError,RuntimeError,KeyError) as e:
        parser.exit(1,str(e)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
