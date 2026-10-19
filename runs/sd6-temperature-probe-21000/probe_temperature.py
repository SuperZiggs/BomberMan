"""SD6.0 probe (validation only, no training): F with lower sampling temperature / argmax.

Schedule E (475/25/40), seed 21000, 400 games per arm, both corners. tau=1.0 is F as evaluated so far.
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = Path(r'D:\fpt\subject\BomberMan'); sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
F = ROOT / 'runs/phaseC-sd-best-fallback/best_phaseC.pt'
TAUS = (1.0, 0.5, 0.25, 0.0)

def run(tau, seed=21000, games=400):
    import torch; torch.set_num_threads(1)
    import evaluate
    from train_actor_critic import ActorCriticNet, smart_load_weights, obs_to_tensor, get_action_mask
    net = ActorCriticNet(in_channels=13, board_size=9); smart_load_weights(net, str(F), 'cpu'); net.eval()
    @torch.no_grad()
    def agent(env, obs, aid):
        logits, _ = net(obs_to_tensor(obs[aid], 'cpu'))
        logits = logits.masked_fill(~torch.tensor(get_action_mask(env, aid), dtype=torch.bool), -1e9)
        if tau == 0.0:
            return int(logits.argmax())
        return int(torch.distributions.Categorical(logits=logits / tau).sample())
    r = evaluate.evaluate_seeded(agent, 9, 2, games, seed, 500, sudden_death=True, sudden_death_start=475,
                                 sudden_death_interval=25, sudden_death_warning_steps=40)
    r['tau'] = tau
    (OUT / f'evaluation_tau{tau}_{seed}.json').write_text(json.dumps(r), encoding='utf-8')
    return r

if __name__ == '__main__':
    with ProcessPoolExecutor(4) as pool:
        results = list(pool.map(run, TAUS))
    from pilot_two_bomb import paired_active_bootstrap
    base = results[0]
    summary = {}
    for r in results:
        row = {k: v for k, v in r.items() if k not in ('games', 'by_corner')}
        if r is not base:
            row['paired_vs_tau1'] = {m: paired_active_bootstrap(*({'games': [{**g, 'active_win': g[m]} for g in x['games']]} for x in (r, base)))
                                     for m in ('active_win', 'win', 'draw', 'suicide')}
        summary[str(r['tau'])] = row
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for t, row in summary.items():
        print(t, {k: row[k] for k in ('win_rate', 'active_win_rate', 'draw_rate', 'suicide_rate', 'sudden_death_death_rate', 'avg_kills', 'avg_steps')},
              {m: (v['difference_pp'], v['ci95_pp']) for m, v in row.get('paired_vs_tau1', {}).items()})
