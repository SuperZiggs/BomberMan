"""Per-decision wall time of S3 (CNN only) vs G + search3, on the same states.

Plays a few validation games (seed 21000.., both corners) with G + search3 as the learner against RuleBot,
and at every learner step times (a) the S3 CNN action and (b) the G + search3 action on that state.
Validation seeds only; no test seed is touched. Single process, CPU, one torch thread.
Usage: python paper/scripts/time_decisions.py [boards]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import evaluate  # noqa: E402
import train_actor_critic as trainer  # noqa: E402
from bomberman_env import BombermanFFAEnv  # noqa: E402
from kill_search import search_action3  # noqa: E402
from rule_based_bot import RuleBasedAgent  # noqa: E402

torch.set_num_threads(1)
G = str(ROOT / 'runs/sd75-20261018/full/best_phaseC.pt')
S3 = str(ROOT / 'runs/sd9-20261023/train/epoch_12.pt')


def main(boards=3):
    g_policy, s3_policy = evaluate.make_ac_fn(G, 9), evaluate.make_ac_fn(S3, 9)
    cnn_ms, search_ms = [], []
    for seed in range(21000, 21000 + boards):
        for aid in (0, 1):
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=seed, sudden_death=True,
                                  sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)
            obs = env.reset()
            bot = RuleBasedAgent(1 - aid)
            for _ in range(500):
                t0 = time.perf_counter()
                s3_policy(env, obs, aid)
                t1 = time.perf_counter()
                mask = trainer.get_action_mask(env, aid)
                action = search_action3(env, aid, mask, g_policy(env, obs, aid))[0]
                t2 = time.perf_counter()
                cnn_ms.append((t1 - t0) * 1e3)
                search_ms.append((t2 - t1) * 1e3)
                actions = [0, 0]
                actions[aid], actions[1 - aid] = action, bot.select_action(env)
                obs, _, _, info = env.step(actions)
                if info.get('episode_done'):
                    break
    cnn, srch = np.array(cnn_ms), np.array(search_ms)
    out = {'decisions': int(len(cnn)), 'games': boards * 2,
           'cnn_ms_mean': float(cnn.mean()), 'cnn_ms_median': float(np.median(cnn)),
           'search3_ms_mean': float(srch.mean()), 'search3_ms_median': float(np.median(srch)),
           'search3_ms_p95': float(np.quantile(srch, 0.95)), 'ratio_of_means': float(srch.mean() / cnn.mean()),
           'torch': torch.__version__, 'threads': 1, 'seeds': [21000, 21000 + boards - 1]}
    assert out['decisions'] > 0
    (ROOT / 'paper' / 'figures' / 'decision_timing.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
