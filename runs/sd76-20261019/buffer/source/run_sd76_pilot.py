"""SD7.6: group RL on "bomb now" moments with a replay buffer, from G (SD7.5 best).

  train --arm control|buffer [--resume]   probe: SD5.1 control config from G, 1000 ep, no validation
  mechanism                               G plays seeds 6850-6899; G, control, buffer (final) replay every
                                          narrow state K=8 x H=20; then control and buffer on validation 21000
  train --arm full [--resume]             only after H1''' passed: buffer config, 3000 ep, validation picks best
  test                                    G vs full best on seed 46000, 400 games, schedule E (gate) and off; once
Rules (docs/superpowers/specs/2026-10-19-sd7-6-bomb-moment-buffer-design.md, fixed before training):
  H1''': buffer - control conversion at narrow states >= +10pp, state-bootstrap CI95 lower > 0,
         replay deaths <= control + 1pp, validation 21000 buffer - control ActiveWR >= -2pp.
  Gate: full - G ActiveWR >= +5pp, CI95 lower > 0, suicide <= +1pp.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from progress import parallel_map
import run_sd5_pilot as sd5
import run_sd7_pilot as sd7
import run_sd74_probe as sd74

ROOT = Path(__file__).resolve().parent
G = ROOT / 'runs/sd75-20261018/full/best_phaseC.pt'
G_SHA256 = 'bb0c775c58161aa1bbdd229b9528170aab174d5aabf8f4a6739b848919cdc7ac'
BUFFER = dict(group_rl=True, group_coef=4.0, group_k=16, group_horizon=20, group_max_states=6, group_gap=4,
              group_select='narrow', group_buffer=1000, group_first_step=True)
MECH_SEED, MECH_GAMES, MECH_K, VALIDATION_SEED, TEST_SEED, TEST_GAMES = 6850, 100, 8, 21000, 46000, 400
SOURCE_FILES = (*sd74.SOURCE_FILES, 'run_sd76_pilot.py')


def train_config(output, arm, smoke=False):
    config = {**json.loads(sd7.CONTROL_CONFIG.read_text(encoding='utf-8')), 'resume_from': str(G),
              **(BUFFER if arm != 'control' else {}), **(sd74.PROBE if arm != 'full' else {}),
              'checkpoint_dir': str(Path(output) / arm)}
    return {**config, **sd7.SMOKE, **({} if arm == 'full' else {'validation_every': 0})} if smoke else config


def train_arm(output, arm, device, smoke=False, resume=False):
    if sd5._hash(G) != G_SHA256:
        raise ValueError('G checkpoint changed')
    registration = {'G_sha256': G_SHA256, 'mechanism_seed': MECH_SEED, 'test_seed': TEST_SEED,
                    'rules': __doc__.split('Rules')[1].strip()}
    sd74.run_arm(Path(output) / arm, {**train_config(output, arm, smoke), 'device': device}, SOURCE_FILES,
                 registration, f'SD7.6 {arm}', resume)


def mechanism_shard(models, seed, games, max_steps, ticks):
    """G plays; up to 6 narrow states per game (>= 4 steps apart); every model replays each one."""
    import torch
    torch.set_num_threads(1)
    from copy import deepcopy
    import evaluate
    import group_rl
    from train_actor_critic import get_action_mask
    nets = {name: sd7._net(path) for name, path in models.items()}
    policy = evaluate.make_ac_fn(str(G), 9)
    rows, game = [], {'env': None, 'kept': 0, 'last': -99}

    def agent(env, obs, aid):
        if game['env'] is not env:
            game.update(env=env, kept=0, last=-99)
        if game['kept'] < BUFFER['group_max_states'] and env.step_count - game['last'] >= BUFFER['group_gap']:
            mask = get_action_mask(env, aid)
            kill = group_rl.kill_actions(env, aid, mask)
            if 1 <= len(kill) <= 2:
                game.update(kept=game['kept'] + 1, last=env.step_count)
                snap, index, row = deepcopy(env), len(rows), {'seed': seed, 'step': env.step_count, 'kill': kill}
                x = torch.from_numpy(np.asarray(obs[aid], dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    row['p_kill'] = {name: float(torch.softmax(net(x)[0].masked_fill(
                        ~torch.tensor(mask), -1e9), -1)[0, kill].sum()) for name, net in nets.items()}
                row['outcomes'] = {name: [group_rl.rollout(
                    snap, aid, group_rl.sampler(net, 'cpu', 100003 * seed + 97 * index + k),
                    100003 * seed + 97 * index + k, BUFFER['group_horizon'])[0] for k in range(MECH_K)]
                    for name, net in nets.items()}
                rows.append(row)
        return policy(env, obs, aid)

    def advance(done, total):
        if done:
            ticks.put(1)
    evaluate.evaluate_seeded(agent, 9, 2, games, seed, max_steps, on_progress=advance,
                             **{'sudden_death': True, **sd5.SCHEDULE_E})
    ticks.put(1)
    return rows


def mechanism_passes(summary):
    vs = summary['buffer_vs_control']
    return (vs['difference_pp'] >= 10.0 and vs['ci95_pp'][0] > 0
            and summary['deaths']['buffer'] <= summary['deaths']['control'] + 0.01
            and summary['validation']['buffer_vs_control']['difference_pp'] >= -2.0)


def mechanism(output, workers=4, smoke=False):
    from pilot_two_bomb import paired_active_bootstrap
    output = Path(output)
    models = {'G': G, 'control': output / 'control' / 'final_phaseC.pt', 'buffer': output / 'buffer' / 'final_phaseC.pt'}
    games, steps, vgames = (2, 60, 2) if smoke else (MECH_GAMES, 500, 400)
    with open(output / 'mechanism_opened.json', 'x', encoding='utf-8') as handle:  # diagnostic seed used once
        json.dump({'seed': MECH_SEED, 'models': {k: sd5._hash(p) for k, p in models.items()}}, handle, indent=2)
    cuts = [games // 2 * i // workers for i in range(workers + 1)]
    jobs = [(models, MECH_SEED + cuts[i], 2 * (cuts[i + 1] - cuts[i]), steps) for i in range(workers) if cuts[i + 1] > cuts[i]]
    rows = [r for part in parallel_map(mechanism_shard, jobs, total=games + len(jobs), desc='SD7.6 mechanism',
                                       workers=workers) for r in part]
    (output / 'mechanism_states.json').write_text(json.dumps(rows), encoding='utf-8')
    arms = ('control', 'buffer')
    validation = dict(zip(arms, parallel_map(sd5.evaluate_model, [(models[a], 'E', VALIDATION_SEED, vgames, steps) for a in arms],
                                             total=2 * vgames, desc='SD7.6 validation 21000', workers=2)))
    for arm, result in validation.items():
        (output / f'validation_{arm}_{VALIDATION_SEED}.json').write_text(json.dumps(result), encoding='utf-8')
    outcomes = {m: [r['outcomes'][m] for r in rows] for m in models}
    summary = {'seed': MECH_SEED, 'games': games, 'states': len(rows),
               'buffer_vs_control': sd7.paired_states(outcomes['buffer'], outcomes['control']),
               'buffer_vs_g': sd7.paired_states(outcomes['buffer'], outcomes['G']),
               'control_vs_g': sd7.paired_states(outcomes['control'], outcomes['G']),
               'deaths': {m: float(np.mean([np.mean(np.array(x) == -1) for x in outcomes[m]])) if rows else 0.0
                          for m in models},
               'p_kill': {m: float(np.mean([r['p_kill'][m] for r in rows])) if rows else None for m in models},
               'validation': {**{a: {k: v for k, v in validation[a].items() if k not in ('games', 'by_corner')} for a in arms},
                              'buffer_vs_control': paired_active_bootstrap(validation['buffer'], validation['control'])}}
    summary['h1_pass'] = mechanism_passes(summary)
    (output / 'mechanism_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def gate(paired):
    vs_g = paired['full_vs_g']
    checks = {'active_vs_g_5pp': vs_g['active_win']['difference_pp'] >= 5.0,
              'active_vs_g_ci_lower_positive': vs_g['active_win']['ci95_pp'][0] > 0,
              'suicide_within_1pp': vs_g['suicide']['difference_pp'] <= 1.0}
    return {**checks, 'pass': all(checks.values())}


def test(output, smoke=False, workers=3):
    from pilot_two_bomb import paired_active_bootstrap
    output = Path(output)
    paths = {'G': G, 'full': output / 'full' / 'best_phaseC.pt'}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f'missing models: {missing}')
    seed, games, steps = (21000, 2, 30) if smoke else (TEST_SEED, TEST_GAMES, 500)
    with open(output / 'test_opened.json', 'x', encoding='utf-8') as handle:  # the test seed is used once
        json.dump({'seed': seed, 'games': games, 'models': {k: sd5._hash(p) for k, p in paths.items()}}, handle, indent=2)
    jobs = [(name, rule) for name in paths for rule in sd7.RULES]
    evaluated = parallel_map(sd5.evaluate_model, [(paths[n], r, seed, games, steps) for n, r in jobs],
                             total=len(jobs) * games, desc=f'SD7.6 test seed {seed}', workers=workers)
    results = {f'{n}_{r}': result for (n, r), result in zip(jobs, evaluated)}
    for key, result in results.items():
        (output / f'evaluation_{key}_{seed}.json').write_text(json.dumps(result), encoding='utf-8')
    paired = {rule: {'full_vs_g': {
        m: paired_active_bootstrap(*({'games': [{**g, 'active_win': g[m]} for g in results[f'{x}_{rule}']['games']]}
                                     for x in ('full', 'G'))) for m in ('active_win', 'win', 'draw', 'suicide')}}
        for rule in sd7.RULES}
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
    p.add_argument('--arm', choices=('control', 'buffer', 'full'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.stage == 'train':
        if not a.arm:
            p.error('train needs --arm')
        train_arm(a.output, a.arm, a.device, a.smoke, a.resume)
    elif a.stage == 'mechanism':
        mechanism(a.output, a.workers, a.smoke)
    else:
        test(a.output, a.smoke, min(a.workers, 3))
