"""SD7.4: mechanism probe for denser group updates; 1000 episodes from F, control vs group.

  train --arm control|group [--resume]   SD5.1 control config, 1000 ep, no validation; output <out>/<arm>
  mechanism                              F, control, group (final) replay fixed decisive states, seeds 6700-6749
Rule (docs/superpowers/specs/2026-10-18-sd7-4-dense-group-probe-design.md, fixed before training):
  H1': group - control conversion >= +5pp, state-bootstrap CI95 lower > 0, replay deaths <= control + 1pp.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from progress import bar, parallel_map
import run_sd5_pilot as sd5
import run_sd7_pilot as sd7

ROOT = Path(__file__).resolve().parent
PROBE = dict(episodes=1000, validation_every=0)
GROUP = dict(group_rl=True, group_coef=4.0, group_k=16, group_horizon=20, group_max_states=6, group_gap=4)
MECH_SEED, MECH_GAMES = 6700, 100
SOURCE_FILES = (*sd7.SOURCE_FILES, 'run_sd74_probe.py')


def train_config(output, arm, smoke=False):
    config = {**json.loads(sd7.CONTROL_CONFIG.read_text(encoding='utf-8')), **PROBE,
              **(GROUP if arm == 'group' else {}), 'checkpoint_dir': str(Path(output) / arm)}
    return {**config, **sd7.SMOKE, 'validation_every': 0} if smoke else config


def train_arm(output, arm, device, smoke=False, resume=False):
    import train_actor_critic as trainer
    directory = Path(output) / arm
    if sd5._hash(sd5.F) != sd5.F_SHA256:
        raise ValueError('F checkpoint changed')
    config = {**train_config(output, arm, smoke), 'device': device}
    state, metrics = directory / 'training_state.pt', directory / 'metrics.jsonl'
    if resume:
        if json.loads((directory / 'config.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Arm config changed; resume refused')
        from training_state import load_training_state
        done = load_training_state(state)['episode']
        rows = [line for line in metrics.read_text(encoding='utf-8').splitlines() if json.loads(line)['episode'] <= done]
        metrics.write_text(''.join(line + '\n' for line in rows), encoding='utf-8')
        config = {**config, 'resume_training_state': str(state)}
    else:
        done = 0
        directory.mkdir(parents=True)  # FileExistsError: never overwrite
        (directory / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        (directory / 'source').mkdir()
        for name in SOURCE_FILES:
            shutil.copy2(ROOT / name, directory / 'source' / name)
        (directory / 'registration.json').write_text(json.dumps(
            {'F_sha256': sd5.F_SHA256, 'sources': {n: sd5._hash(ROOT / n) for n in SOURCE_FILES},
             'mechanism_seed': MECH_SEED, 'rules': __doc__.split('Rule')[1].strip()}, indent=2), encoding='utf-8')
    recent = []
    with open(metrics, 'a', encoding='utf-8') as handle, bar(config['episodes'], f'SD7.4 {arm}', done) as shown:
        def write(row):
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            recent[:] = (recent + [row['active_win']])[-100:]
            g = row.get('group')
            shown.update(1)
            shown.set_postfix_str(f"ActiveWR(100) {100 * sum(recent) / len(recent):.0f}%"
                                  + (f" states {g['states']} wins {g['wins']}/{g['rollouts']}" if g else ''))
        trainer.train(**{k: v for k, v in config.items() if k != 'device'}, device=device, on_episode=write)


def probe_passes(summary):
    vs = summary['group_vs_control']
    return (vs['difference_pp'] >= 5.0 and vs['ci95_pp'][0] > 0
            and summary['deaths']['group'] <= summary['deaths']['control'] + 0.01)


def mechanism(output, workers=4, smoke=False):
    output = Path(output)
    models = {'F': sd5.F, 'control': output / 'control' / 'final_phaseC.pt', 'group': output / 'group' / 'final_phaseC.pt'}
    games, steps = (2, 60) if smoke else (MECH_GAMES, 500)
    cuts = [games // 2 * i // workers for i in range(workers + 1)]
    jobs = [(models, MECH_SEED + cuts[i], 2 * (cuts[i + 1] - cuts[i]), steps) for i in range(workers) if cuts[i + 1] > cuts[i]]
    rows = [r for part in parallel_map(sd7.mechanism_shard, jobs, total=games, desc='SD7.4 mechanism', workers=workers)
            for r in part]
    (output / 'mechanism_states.json').write_text(json.dumps(rows), encoding='utf-8')
    outcomes = {m: [r['outcomes'][m] for r in rows] for m in sd7.MODELS}
    summary = {'seed': MECH_SEED, 'games': games, 'states': len(rows),
               'models': {k: sd5._hash(p) for k, p in models.items()},
               'group_vs_control': sd7.paired_states(outcomes['group'], outcomes['control']),
               'group_vs_f': sd7.paired_states(outcomes['group'], outcomes['F']),
               'control_vs_f': sd7.paired_states(outcomes['control'], outcomes['F']),
               'deaths': {m: float(np.mean([np.mean(np.array(x) == -1) for x in outcomes[m]])) for m in sd7.MODELS}}
    summary['h1_pass'] = probe_passes(summary)
    (output / 'mechanism_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    import torch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('train', 'mechanism'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--arm', choices=('control', 'group'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.stage == 'train':
        if not a.arm:
            p.error('train needs --arm')
        train_arm(a.output, a.arm, a.device, a.smoke, a.resume)
    else:
        mechanism(a.output, a.workers, a.smoke)
