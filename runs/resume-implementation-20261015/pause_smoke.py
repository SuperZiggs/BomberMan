import sys
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import run_reward_pilot as runner
import train_actor_critic as trainer
original = trainer.train

def pause(**kwargs):
    callback = kwargs['on_episode']
    def on_episode(row):
        callback(row)
        if kwargs['wall_destroy_bonus'] == .075 and row['episode'] == 10:
            kwargs['stop_event'].set()
    kwargs['on_episode'] = on_episode
    return original(**kwargs)

with patch.object(trainer, 'train', side_effect=pause):
    result = runner.run((ROOT / 'runs/reward-resume-smoke-cpu-20261015').resolve(), 'cpu', smoke=True)
assert result['status'] == 'incomplete'
assert result['arms']['control']['completed_episodes'] == 20
assert result['arms']['half_wall']['completed_episodes'] == 10
assert not (ROOT / 'runs/reward-resume-smoke-cpu-20261015/test_opened.json').exists()
