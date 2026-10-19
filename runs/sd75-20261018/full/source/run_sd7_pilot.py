"""SD7: F + group-relative updates from decisive states, 3000 episodes; control = SD5.1 control.

  train [--resume]   SD5.1 control config + group_rl (group_rl.py); output <out>/group
  mechanism          H1: fixed decisive states from F's games (seeds 6650-6699, 100 games);
                     F, control and group each replay every state K=8 x H=20 with the same seeds
  test               F, control best, group best on seed 44000, 400 games each, schedule E (gate)
                     and sudden death off (reported); runs once
Rules (docs/superpowers/specs/2026-10-18-sd7-group-advantage-design.md, fixed before training):
  H1: group - F conversion >= +10pp with state-bootstrap CI95 lower > 0.
  Gate: group - F ActiveWR >= +5pp, CI95 lower > 0, group - control >= 0, suicide <= +1pp.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from progress import bar, parallel_map
import run_sd5_pilot as sd5

ROOT = Path(__file__).resolve().parent
F, F_SHA256, SCHEDULE_E = sd5.F, sd5.F_SHA256, sd5.SCHEDULE_E
CONTROL_DIR = ROOT / 'runs/sd5-20261017/control'
CONTROL_CONFIG, CONTROL_BEST = CONTROL_DIR / 'config.json', CONTROL_DIR / 'best_phaseC.pt'
CONTROL_SHA256 = 'c3b4e37caa0e5bc254984157aed93233d25d44043d2778b54d8f55ba91bfd899'
GROUP = dict(group_rl=True, group_coef=1.0, group_k=8, group_horizon=20, group_max_states=3, group_gap=8)
MODELS, RULES = ('F', 'control', 'group'), ('E', 'off')
TEST_SEED, TEST_GAMES, MECH_SEED, MECH_GAMES = 44000, 400, 6650, 100
SMOKE = dict(episodes=3, max_steps=60, validation_every=2, validation_episodes=2, log_every=1)
SOURCE_FILES = (*sd5.SOURCE_FILES, 'group_rl.py', 'run_sd7_pilot.py')


def train_config(output, smoke=False):
    config = {**json.loads(CONTROL_CONFIG.read_text(encoding='utf-8')), **GROUP,
              'checkpoint_dir': str(Path(output) / 'group')}
    return {**config, **SMOKE} if smoke else config


def train_arm(output, device, smoke=False, resume=False):
    import train_actor_critic as trainer
    directory = Path(output) / 'group'
    if sd5._hash(F) != F_SHA256:
        raise ValueError('F checkpoint changed')
    config = {**train_config(output, smoke), 'device': device}
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
            {'F_sha256': F_SHA256, 'control': str(CONTROL_BEST), 'control_sha256': CONTROL_SHA256,
             'sources': {n: sd5._hash(ROOT / n) for n in SOURCE_FILES}, 'test_seed': TEST_SEED,
             'rules': __doc__.split('Rules')[1].strip()}, indent=2), encoding='utf-8')
    recent = []
    with open(metrics, 'a', encoding='utf-8') as handle, bar(config['episodes'], 'SD7 group', done) as shown:
        def write(row):
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            recent[:] = (recent + [row['active_win']])[-100:]
            g = row['group']
            shown.update(1)
            shown.set_postfix_str(f"ActiveWR(100) {100 * sum(recent) / len(recent):.0f}% best ep {row['best_episode']}"
                                  f" states {g['states']} wins {g['wins']}/{g['rollouts']}")

        def validation(row):
            if 'completed' in row:
                shown.set_postfix_str(f"validation ep {row['episode']}: {row['completed']}/{row['total']} games")
        trainer.train(**{k: v for k, v in config.items() if k != 'device'}, device=device,
                      on_episode=write, on_validation=validation)


def _net(path):
    from train_actor_critic import ActorCriticNet, smart_load_weights
    net = ActorCriticNet(in_channels=13, board_size=9)
    smart_load_weights(net, str(path), 'cpu')
    return net.eval()


def mechanism_shard(models, seed, games, max_steps, ticks):
    """F plays; up to 3 decisive states per game (>= 8 steps apart); every model replays each one."""
    import torch
    torch.set_num_threads(1)
    from copy import deepcopy
    import evaluate
    import group_rl
    from train_actor_critic import get_action_mask
    nets = {name: _net(path) for name, path in models.items()}
    policy = evaluate.make_ac_fn(str(F), 9)
    rows, game = [], {'env': None, 'kept': 0, 'last': -99}

    def agent(env, obs, aid):
        if game['env'] is not env:
            game.update(env=env, kept=0, last=-99)
        if (game['kept'] < GROUP['group_max_states'] and env.step_count - game['last'] >= GROUP['group_gap']
                and group_rl.decisive(env, aid, get_action_mask(env, aid))):
            game.update(kept=game['kept'] + 1, last=env.step_count)
            snap, index = deepcopy(env), len(rows)
            rows.append({'seed': seed, 'step': env.step_count, 'outcomes': {
                name: [group_rl.rollout(snap, aid, group_rl.sampler(net, 'cpu', 100003 * seed + 97 * index + k),
                                        100003 * seed + 97 * index + k, GROUP['group_horizon'])[0]
                       for k in range(GROUP['group_k'])] for name, net in nets.items()}})
        return policy(env, obs, aid)

    def advance(done, total):
        if done:
            ticks.put(1)
    evaluate.evaluate_seeded(agent, 9, 2, games, seed, max_steps, on_progress=advance, **{'sudden_death': True, **SCHEDULE_E})
    ticks.put(1)
    return rows


def paired_states(left, right, draws=10000, seed=0):
    """Win rate (pp) of left - right over replays, bootstrapped over states."""
    l = np.array([np.mean(np.array(x) == 1) for x in left])
    r = np.array([np.mean(np.array(x) == 1) for x in right])
    d = l - r
    idx = np.random.default_rng(seed).integers(0, len(d), (draws, len(d)))
    boots = d[idx].mean(axis=1)
    return {'difference_pp': 100 * float(d.mean()), 'ci95_pp': [100 * float(np.percentile(boots, 2.5)),
                                                                 100 * float(np.percentile(boots, 97.5))],
            'states': int(len(d)), 'left_pp': 100 * float(l.mean()), 'right_pp': 100 * float(r.mean())}


def mechanism_passes(paired):
    return paired['difference_pp'] >= 10.0 and paired['ci95_pp'][0] > 0


def mechanism(output, workers=4, smoke=False):
    output = Path(output)
    models = {'F': F, 'control': CONTROL_BEST, 'group': output / 'group' / 'best_phaseC.pt'}
    games, steps = (2, 60) if smoke else (MECH_GAMES, 500)
    cuts = [games // 2 * i // workers for i in range(workers + 1)]
    jobs = [(models, MECH_SEED + cuts[i], 2 * (cuts[i + 1] - cuts[i]), steps) for i in range(workers) if cuts[i + 1] > cuts[i]]
    rows = [r for part in parallel_map(mechanism_shard, jobs, total=games, desc='SD7 mechanism', workers=workers) for r in part]
    (output / 'mechanism_states.json').write_text(json.dumps(rows), encoding='utf-8')
    outcomes = {m: [r['outcomes'][m] for r in rows] for m in MODELS}
    summary = {'seed': MECH_SEED, 'games': games, 'states': len(rows),
               'group_vs_f': paired_states(outcomes['group'], outcomes['F']),
               'control_vs_f': paired_states(outcomes['control'], outcomes['F']),
               'group_vs_control': paired_states(outcomes['group'], outcomes['control']),
               'deaths': {m: float(np.mean([np.mean(np.array(x) == -1) for x in outcomes[m]])) for m in MODELS}}
    summary['h1_pass'] = mechanism_passes(summary['group_vs_f'])
    (output / 'mechanism_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def gate(paired):
    vs_f, vs_control = paired['group_vs_f'], paired['group_vs_control']
    checks = {'active_vs_f_5pp': vs_f['active_win']['difference_pp'] >= 5.0,
              'active_vs_f_ci_lower_positive': vs_f['active_win']['ci95_pp'][0] > 0,
              'not_worse_than_control': vs_control['active_win']['difference_pp'] >= 0,
              'suicide_within_1pp': vs_f['suicide']['difference_pp'] <= 1.0}
    return {**checks, 'pass': all(checks.values())}


def test(output, smoke=False, workers=3, seed=TEST_SEED, group='group'):
    from pilot_two_bomb import paired_active_bootstrap
    output = Path(output)
    paths = {'F': F, 'control': CONTROL_BEST, 'group': output / group / 'best_phaseC.pt'}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f'missing models: {missing}')
    seed, games, steps = (21000, 2, 30) if smoke else (seed, TEST_GAMES, 500)
    with open(output / 'test_opened.json', 'x', encoding='utf-8') as handle:  # the test seed is used once
        json.dump({'seed': seed, 'games': games, 'models': {k: sd5._hash(p) for k, p in paths.items()}}, handle, indent=2)
    jobs = [(name, rule) for name in MODELS for rule in RULES]
    evaluated = parallel_map(sd5.evaluate_model, [(paths[n], r, seed, games, steps) for n, r in jobs],
                             total=len(jobs) * games, desc=f'SD7 test seed {seed}', workers=workers)
    results = {f'{n}_{r}': result for (n, r), result in zip(jobs, evaluated)}
    for key, result in results.items():
        (output / f'evaluation_{key}_{seed}.json').write_text(json.dumps(result), encoding='utf-8')
    paired = {}
    for rule in RULES:
        for left, right in (('group', 'F'), ('control', 'F'), ('group', 'control')):
            paired.setdefault(rule, {})[f'{left}_vs_{right.lower()}'] = {
                m: paired_active_bootstrap(*({'games': [{**g, 'active_win': g[m]} for g in results[f'{x}_{rule}']['games']]}
                                             for x in (left, right))) for m in ('active_win', 'win', 'draw', 'suicide')}
    summary = {'seed': seed, 'games': games,
               'results': {k: {m: v for m, v in r.items() if m not in ('games', 'by_corner')} for k, r in results.items()},
               'paired': paired, 'gate': gate(paired['E'])}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not smoke:
        shutil.make_archive(str(output) + '-send', 'zip', output, '.')
    print(json.dumps({'gate': summary['gate'], 'results': summary['results']}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    import torch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('train', 'mechanism', 'test'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.stage == 'train':
        if sd5._hash(CONTROL_BEST) != CONTROL_SHA256:
            raise ValueError('control checkpoint changed')
        train_arm(a.output, a.device, a.smoke, a.resume)
    elif a.stage == 'mechanism':
        mechanism(a.output, a.workers, a.smoke)
    else:
        test(a.output, a.smoke, min(a.workers, 3))
