"""SD5.1: train F for 3000 more episodes with and without kill-search labels, then test once.

  train --arm control|search   friend's recipe from F (schedule E in 75% of episodes); `search`
                               adds a combat label whenever kill_search finds a strictly better
                               line than the sampled action. Exact resume with --resume.
  test                         F, control best, search best on seed 42000, 400 games each,
                               schedule E (gate) and sudden death off (reported); runs once.
Gate (PLAN_SUDDEN_DEATH.md SD5.1, fixed before training): search - F ActiveWR >= +5pp with
CI95 lower > 0, search - control >= 0, suicide no more than +1pp over F (all under schedule E).
"""
import argparse
import json
from pathlib import Path
import shutil

from progress import bar, parallel_map

ROOT = Path(__file__).resolve().parent
F = ROOT / 'runs/phaseC-sd-best-fallback/best_phaseC.pt'
F_SHA256 = '0315a56f09079333d936db5f55587093f1b6415ed6355d2ef6ae2ab63f8b1483'
ARMS = ('control', 'search')
MODELS = ('F', *ARMS)
RULES = ('E', 'off')
SCHEDULE_E = dict(sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)
TRAIN_SEED, VALIDATION_SEED, TEST_SEED, TEST_GAMES = 2026101701, 21000, 42000, 400
SMOKE = dict(episodes=4, max_steps=30, validation_every=2, validation_episodes=2, log_every=1)
SOURCE_FILES = ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py', 'evaluate.py',
                'training_state.py', 'tactical_teacher.py', 'combat_move_teacher.py', 'kill_search.py',
                'pilot_two_bomb.py', 'run_sd5_pilot.py')


def train_config(arm, output, device, smoke=False):
    config = dict(phase='C', episodes=3000, board_size=9, num_agents=2, max_steps=500, lr=2.5e-5,
                  entropy_coef=0.0075, wall_destroy_bonus=0.15, attack_lookahead=2, min_pressure_gain=0.15,
                  trap_sequence_depth=2, sequence_pressure_gain=0.35, ring_safety_coef=0.6, late_game_start=350,
                  late_game_attack_multiplier=1.75, escape_loss_coef=0.5, combat_loss_coef=0.9,
                  navigation_loss_coef=0.4, sudden_death_probability=0.75, **SCHEDULE_E, seed=TRAIN_SEED,
                  resume_from=str(F), validation_every=500, validation_episodes=400,
                  validation_seed=VALIDATION_SEED, log_every=50, device=device, run_report=False,
                  save_training_state=True, search_teacher=arm == 'search', search_loss_coef=0.9,
                  checkpoint_dir=str(Path(output) / arm))
    return {**config, **SMOKE} if smoke else config


def _hash(path):
    from audit_t2 import digest
    return digest(Path(path))


def train_arm(output, arm, device, smoke=False, resume=False):
    import train_actor_critic as trainer
    output, directory = Path(output), Path(output) / arm
    if _hash(F) != F_SHA256:
        raise ValueError('F checkpoint changed')
    config = train_config(arm, output, device, smoke)
    state, metrics = directory / 'training_state.pt', directory / 'metrics.jsonl'
    if resume:
        if json.loads((directory / 'config.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Arm config changed; resume refused')
        from training_state import load_training_state
        done = load_training_state(state)['episode']
        # Drop rows logged after the last saved state; those episodes are replayed exactly.
        rows = [line for line in metrics.read_text(encoding='utf-8').splitlines() if json.loads(line)['episode'] <= done]
        metrics.write_text(''.join(line + '\n' for line in rows), encoding='utf-8')
        config = {**config, 'resume_training_state': str(state)}
    else:
        done = 0
        directory.mkdir(parents=True)  # FileExistsError: never overwrite an arm
        (directory / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        source = directory / 'source'  # per arm: both arms may start at the same time
        source.mkdir()
        for name in SOURCE_FILES:
            shutil.copy2(ROOT / name, source / name)
        (directory / 'registration.json').write_text(json.dumps(
            {'F_sha256': F_SHA256, 'sources': {n: _hash(ROOT / n) for n in SOURCE_FILES},
             'test_seed': TEST_SEED, 'gate': __doc__.split('Gate')[1].strip()}, indent=2), encoding='utf-8')
    recent = []
    with open(metrics, 'a', encoding='utf-8') as handle, bar(config['episodes'], f'SD5.1 {arm}', done) as shown:
        def write(row):
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            recent[:] = (recent + [row['active_win']])[-100:]
            shown.update(1)
            shown.set_postfix_str(f"ActiveWR(100) {100 * sum(recent) / len(recent):.0f}% best ep {row['best_episode']}"
                                  + (f" search labels {row['search']['labels']}" if 'search' in row else ''))

        def validation(row):
            if 'completed' in row:
                shown.set_postfix_str(f"validation ep {row['episode']}: {row['completed']}/{row['total']} games")
        trainer.train(**config, on_episode=write, on_validation=validation)


def evaluate_model(path, rule, seed, games, max_steps, ticks):
    import torch
    torch.set_num_threads(1)
    import evaluate
    finished = [0]

    def advance(done, total):  # called before each game and once at the end
        ticks.put(done - finished[0])
        finished[0] = done
    return evaluate.evaluate_seeded(evaluate.make_ac_fn(str(path), 9), 9, 2, games, seed, max_steps,
                                    on_progress=advance, sudden_death=rule == 'E', **SCHEDULE_E)


def gate(paired):
    vs_f, vs_control = paired['search_vs_f'], paired['search_vs_control']
    checks = {'active_vs_f_5pp': vs_f['active_win']['difference_pp'] >= 5.0,
              'active_vs_f_ci_lower_positive': vs_f['active_win']['ci95_pp'][0] > 0,
              'not_worse_than_control': vs_control['active_win']['difference_pp'] >= 0,
              'suicide_within_1pp': vs_f['suicide']['difference_pp'] <= 1.0}
    return {**checks, 'pass': all(checks.values())}


def test(output, smoke=False, workers=3):
    from pilot_two_bomb import paired_active_bootstrap
    output = Path(output)
    paths = {'F': F, **{arm: output / arm / 'best_phaseC.pt' for arm in ARMS}}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f'missing models: {missing}')
    opened = output / 'test_opened.json'
    seed, games, steps = (VALIDATION_SEED, 2, 30) if smoke else (TEST_SEED, TEST_GAMES, 500)
    with open(opened, 'x', encoding='utf-8') as handle:  # FileExistsError: the test seed is used once
        json.dump({'seed': seed, 'games': games, 'models': {k: _hash(p) for k, p in paths.items()}}, handle, indent=2)
    jobs = [(name, rule) for name in MODELS for rule in RULES]
    evaluated = parallel_map(evaluate_model, [(paths[n], r, seed, games, steps) for n, r in jobs],
                             total=len(jobs) * games, desc=f'SD5.1 test seed {seed}', workers=workers)
    results = {f'{n}_{r}': result for (n, r), result in zip(jobs, evaluated)}
    for key, result in results.items():
        (output / f'evaluation_{key}_{seed}.json').write_text(json.dumps(result), encoding='utf-8')
    paired = {}
    for rule in RULES:
        for left, right in (('search', 'F'), ('control', 'F'), ('search', 'control')):
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
    p.add_argument('stage', choices=('train', 'test'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    if a.stage == 'train':
        if a.arm is None:
            p.error('train needs --arm')
        train_arm(a.output, a.arm, a.device, a.smoke, a.resume)
    else:
        test(a.output, a.smoke, a.workers)
