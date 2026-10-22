"""S2-KL retention pilot: fixed budgets, registered inputs and resumable stages."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import time

from s2_kl_protocol import ROOT, config_for, seed_manifest, register, verify_run
from s2_kl_training import build_train_schedule, candidates_for_run, train_arm
from s2_kl_evaluation import (load_samples, decision_targets, diagnose_model, validation_tasks,
    cluster_comparison, decide_gate, build_report)
from s2_kl_ledger import diagnose_run, validate_run, package_run, process_peak_rss


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
        started = time.perf_counter()
        if args.stage == 'register':
            result = register(args.output, args.input_root, device=args.device,
                workers=args.workers, smoke=args.smoke, resume=args.resume)
        else:
            verify_run(args.output)
            frozen = args.output.resolve() / 'sources/run_s2_kl_probe.py'
            if Path(__file__).resolve() != frozen:
                import subprocess
                return subprocess.call([sys.executable, str(frozen), *(argv if argv is not None else sys.argv[1:])])
            if args.stage == 'train':
                result = train_arm(args.output, args.arm, resume=args.resume)
            elif args.stage == 'diagnose':
                result = diagnose_run(args.output, resume=args.resume)
            elif args.stage == 'validate':
                result = validate_run(args.output, resume=args.resume)
            else:
                result = package_run(args.output)
        measurement = dict(stage=args.stage, output=str(args.output), status=result.get('status', 'OK'),
            elapsed_seconds=time.perf_counter() - started, peak_rss_bytes=process_peak_rss(), pid=os.getpid())
        from training_state import run_lock
        with run_lock(args.output.resolve()):
            path = args.output.resolve() / 'logs/stages.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(measurement, allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
        print(json.dumps(measurement))
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
