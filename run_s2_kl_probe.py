"""S2-KL retention pilot: fixed budgets, registered inputs and resumable stages."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys

from s2_kl_protocol import ROOT, config_for, seed_manifest, register, verify_run
from s2_kl_training import build_train_schedule, candidates_for_run, train_arm
from s2_kl_evaluation import (load_samples, decision_targets, diagnose_model, validation_tasks,
    cluster_comparison, decide_gate, build_report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    for stage in ('register', 'train', 'diagnose', 'validate', 'package'):
        p = sub.add_parser(stage)
        p.add_argument('--output', type=Path, required=True)
        if stage != 'package':
            p.add_argument('--resume', action='store_true')
        if stage == 'register':
            p.add_argument('--input-root', type=Path, default=ROOT)
            p.add_argument('--device', choices=('cpu', 'cuda', 'cuda:0'), default='cuda:0')
            p.add_argument('--workers', type=int, choices=(1, 2), default=2)
            p.add_argument('--smoke', action='store_true')
        if stage == 'train':
            p.add_argument('--arm', choices=('control', 'anchor'), required=True)
    try:
        args = parser.parse_args(argv)
        if args.stage == 'register':
            result = register(args.output, args.input_root, device=args.device,
                workers=args.workers, smoke=args.smoke, resume=args.resume)
        else:
            verify_run(args.output)
            raise ValueError('Stage is not implemented yet: ' + args.stage)
        print(json.dumps({'stage': args.stage, 'output': str(args.output),
            'status': result.get('status', 'OK')}))
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
