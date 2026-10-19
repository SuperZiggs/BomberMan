"""SD7.7: SD7.6 bomb-moment buffer + KL anchor to G (stops "always bomb when legal").

  calibrate                         4 short runs (150 ep) from G, beta in {0, 1, 4, 16}; measured on 40 G games,
                                    seed 6600; picks beta by the fixed rule -> <out>/calib/summary.json
  train --arm anchor [--resume]     probe: SD7.6 buffer config + chosen beta, 1000 ep, no validation
  mechanism                         seeds 6900-6949: G, SD7.6 control, anchor replay narrow states; anchor on validation 21000
  train --arm full [--resume]       only after H1'''' passed: 3000 ep, validation picks best
  test                              G vs full best on seed 46000 (run_sd76_pilot.test), once
Rules (docs/superpowers/specs/2026-10-19-sd7-7-kl-anchor-design.md, fixed before running):
  beta: among beta > 0 with P(bomb | legal) <= P_G + 0.05, the highest P(kill) at narrow states (ties: smaller beta).
  H1'''': anchor - control conversion >= +10pp, CI95 lower > 0, replay deaths <= control + 1pp; validation 21000:
          kills/game >= control, bombs/game <= control * 1.10, ActiveWR anchor - control >= 0.
  Gate: full - G ActiveWR >= +5pp, CI95 lower > 0, suicide <= +1pp.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from progress import parallel_map
import run_sd5_pilot as sd5
import run_sd7_pilot as sd7
import run_sd74_probe as sd74
import run_sd76_pilot as sd76

ROOT = Path(__file__).resolve().parent
BETAS, CALIB_EPISODES, CALIB_SEED, CALIB_GAMES = (0.0, 1.0, 4.0, 16.0), 150, 6600, 40
MECH_SEED, MECH_GAMES = 6900, 100
SD76 = ROOT / 'runs/sd76-20261019'
CONTROL, CONTROL_VALIDATION = SD76 / 'control/final_phaseC.pt', SD76 / 'validation_control_21000.json'
SOURCE_FILES = (*sd76.SOURCE_FILES, 'run_sd77_pilot.py')


def train_config(output, arm, beta, smoke=False):
    base = sd76.train_config(output, 'full' if arm == 'full' else 'buffer', smoke)
    return {**base, 'anchor_coef': beta, 'checkpoint_dir': str(Path(output) / arm)}


def chosen_beta(output):
    beta = json.loads((Path(output) / 'calib' / 'summary.json').read_text(encoding='utf-8'))['chosen_beta']
    if beta is None:
        raise ValueError('calibration chose no beta; see calib/summary.json')
    return beta


def train_arm(output, arm, device, smoke=False, resume=False):
    if sd5._hash(sd76.G) != sd76.G_SHA256:
        raise ValueError('G checkpoint changed')
    registration = {'G_sha256': sd76.G_SHA256, 'mechanism_seed': MECH_SEED, 'test_seed': sd76.TEST_SEED,
                    'rules': __doc__.split('Rules')[1].strip()}
    sd74.run_arm(Path(output) / arm, {**train_config(output, arm, chosen_beta(output), smoke), 'device': device},
                 SOURCE_FILES, registration, f'SD7.7 {arm}', resume)


def calib_train(output, beta, device, smoke, ticks):
    import torch
    torch.set_num_threads(1)
    import train_actor_critic as trainer
    config = {**train_config(output, 'anchor', beta, smoke), 'checkpoint_dir': str(Path(output) / 'calib' / f'beta_{beta:g}')}
    if not smoke:
        config['episodes'] = CALIB_EPISODES
    Path(config['checkpoint_dir']).mkdir(parents=True)  # FileExistsError: never overwrite
    (Path(config['checkpoint_dir']) / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    trainer.train(**{k: v for k, v in config.items() if k != 'device'}, device=device, on_episode=lambda row: ticks.put(1))
    return config['checkpoint_dir']


def measure_shard(models, seed, games, max_steps, ticks):
    """G plays; per model: P(bomb | bomb legal), KL(model || G) over learner steps, P(kill set) at narrow states."""
    import torch
    torch.set_num_threads(1)
    import evaluate
    import group_rl
    from bomberman_env import ACTION_BOMB
    from train_actor_critic import get_action_mask
    nets = {name: sd7._net(path) for name, path in models.items()}
    policy = evaluate.make_ac_fn(str(sd76.G), 9)
    out = {name: {'bomb': [], 'kl': [], 'kill': []} for name in nets}

    def agent(env, obs, aid):
        mask = get_action_mask(env, aid)
        m = torch.tensor(mask)
        x = torch.from_numpy(np.asarray(obs[aid], dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            logp = {k: torch.log_softmax(n(x)[0].masked_fill(~m, -1e9), -1)[0] for k, n in nets.items()}
        kill = group_rl.kill_actions(env, aid, mask)
        for k, lp in logp.items():
            out[k]['kl'].append(float((lp.exp() * (lp - logp['G'])).sum()))
            if mask[ACTION_BOMB]:
                out[k]['bomb'].append(float(lp[ACTION_BOMB].exp()))
            if 1 <= len(kill) <= 2:
                out[k]['kill'].append(float(lp[kill].exp().sum()))
        return policy(env, obs, aid)

    def advance(done, total):
        if done:
            ticks.put(1)
    evaluate.evaluate_seeded(agent, 9, 2, games, seed, max_steps, on_progress=advance,
                             **{'sudden_death': True, **sd5.SCHEDULE_E})
    ticks.put(1)
    return out


def choose(summary):
    """The fixed beta rule; summary[name] = {'p_bomb', 'p_kill'} with names 'G' and 'beta_<b>'."""
    limit = summary['G']['p_bomb'] + 0.05
    ok = [b for b in BETAS if b > 0 and summary[f'beta_{b:g}']['p_bomb'] <= limit]
    return max(ok, key=lambda b: (summary[f'beta_{b:g}']['p_kill'], -b)) if ok else None


def calibrate(output, device, workers=4, smoke=False):
    output = Path(output)
    dirs = parallel_map(calib_train, [(output, b, device, smoke) for b in BETAS],
                        total=len(BETAS) * (sd7.SMOKE['episodes'] if smoke else CALIB_EPISODES),
                        desc='SD7.7 calibrate train', workers=min(workers, len(BETAS)))
    models = {'G': sd76.G, **{Path(d).name: Path(d) / 'final_phaseC.pt' for d in dirs}}
    games, steps = (2, 60) if smoke else (CALIB_GAMES, 500)
    cuts = [games // 2 * i // workers for i in range(workers + 1)]
    jobs = [(models, CALIB_SEED + cuts[i], 2 * (cuts[i + 1] - cuts[i]), steps) for i in range(workers) if cuts[i + 1] > cuts[i]]
    parts = parallel_map(measure_shard, jobs, total=games + len(jobs), desc='SD7.7 calibrate measure', workers=workers)
    summary = {name: {'p_bomb': float(np.mean([v for p in parts for v in p[name]['bomb']])),
                      'kl_to_g': float(np.mean([v for p in parts for v in p[name]['kl']])),
                      'p_kill': float(np.mean([v for p in parts for v in p[name]['kill']])) if any(p[name]['kill'] for p in parts) else 0.0,
                      'narrow_states': sum(len(p[name]['kill']) for p in parts)} for name in models}
    summary['chosen_beta'] = choose(summary)
    summary['beta0_spilled'] = summary['beta_0']['p_bomb'] > summary['G']['p_bomb'] + 0.05
    (output / 'calib' / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def mechanism_passes(summary):
    vs, val = summary['anchor_vs_control'], summary['validation']
    return (vs['difference_pp'] >= 10.0 and vs['ci95_pp'][0] > 0
            and summary['deaths']['anchor'] <= summary['deaths']['control'] + 0.01
            and val['anchor']['avg_kills'] >= val['control']['avg_kills']
            and val['anchor']['avg_bombs'] <= val['control']['avg_bombs'] * 1.10
            and val['anchor_vs_control']['difference_pp'] >= 0)


def mechanism(output, workers=4, smoke=False):
    from pilot_two_bomb import paired_active_bootstrap
    output = Path(output)
    models = {'G': sd76.G, 'control': CONTROL, 'anchor': output / 'anchor' / 'final_phaseC.pt'}
    games, steps, vgames = (2, 60, 2) if smoke else (MECH_GAMES, 500, 400)
    with open(output / 'mechanism_opened.json', 'x', encoding='utf-8') as handle:  # diagnostic seed used once
        json.dump({'seed': MECH_SEED, 'models': {k: sd5._hash(p) for k, p in models.items()}}, handle, indent=2)
    cuts = [games // 2 * i // workers for i in range(workers + 1)]
    jobs = [(models, MECH_SEED + cuts[i], 2 * (cuts[i + 1] - cuts[i]), steps) for i in range(workers) if cuts[i + 1] > cuts[i]]
    rows = [r for part in parallel_map(sd76.mechanism_shard, jobs, total=games + len(jobs), desc='SD7.7 mechanism',
                                       workers=workers) for r in part]
    (output / 'mechanism_states.json').write_text(json.dumps(rows), encoding='utf-8')
    anchor = parallel_map(sd5.evaluate_model, [(models['anchor'], 'E', sd76.VALIDATION_SEED, vgames, steps)],
                          total=vgames, desc='SD7.7 validation 21000', workers=1)[0]
    (output / f'validation_anchor_{sd76.VALIDATION_SEED}.json').write_text(json.dumps(anchor), encoding='utf-8')
    control = json.loads(CONTROL_VALIDATION.read_text(encoding='utf-8'))
    if smoke:  # the stored control result has 400 games; pair against a same-size run instead
        control = parallel_map(sd5.evaluate_model, [(CONTROL, 'E', sd76.VALIDATION_SEED, vgames, steps)],
                               total=vgames, desc='SD7.7 smoke control', workers=1)[0]
    outcomes = {m: [r['outcomes'][m] for r in rows] for m in models}
    strip = lambda r: {k: v for k, v in r.items() if k not in ('games', 'by_corner')}
    summary = {'seed': MECH_SEED, 'games': games, 'states': len(rows),
               'anchor_vs_control': sd7.paired_states(outcomes['anchor'], outcomes['control']),
               'anchor_vs_g': sd7.paired_states(outcomes['anchor'], outcomes['G']),
               'control_vs_g': sd7.paired_states(outcomes['control'], outcomes['G']),
               'deaths': {m: float(np.mean([np.mean(np.array(x) == -1) for x in outcomes[m]])) if rows else 0.0
                          for m in models},
               'p_kill': {m: float(np.mean([r['p_kill'][m] for r in rows])) if rows else None for m in models},
               'validation': {'anchor': strip(anchor), 'control': strip(control),
                              'anchor_vs_control': paired_active_bootstrap(anchor, control)}}
    summary['h1_pass'] = mechanism_passes(summary)
    (output / 'mechanism_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    import torch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('calibrate', 'train', 'mechanism', 'test'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--arm', choices=('anchor', 'full'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.stage == 'calibrate':
        calibrate(a.output, a.device, a.workers, a.smoke)
    elif a.stage == 'train':
        if not a.arm:
            p.error('train needs --arm')
        train_arm(a.output, a.arm, a.device, a.smoke, a.resume)
    elif a.stage == 'mechanism':
        mechanism(a.output, a.workers, a.smoke)
    else:
        sd76.test(a.output, a.smoke, min(a.workers, 3))
