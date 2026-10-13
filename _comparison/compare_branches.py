"""Reproducible branch comparison; writes only to this comparison directory."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / "_comparison" / "BomberMan-Loc_update"
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import bomberman_env as local_env
import train_actor_critic as local_train
from evaluate import evaluate_seeded, make_ac_fn

torch.set_num_threads(1)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_remote():
    original = {name: sys.modules.get(name) for name in
                ("bomberman_env", "rule_based_bot", "train_actor_critic")}
    try:
        env = load_module("_loc_env", REMOTE / "bomberman_env.py")
        sys.modules["bomberman_env"] = env
        bot = load_module("_loc_bot", REMOTE / "rule_based_bot.py")
        sys.modules["rule_based_bot"] = bot
        train = load_module("_loc_train", REMOTE / "train_actor_critic.py")
        return env, train
    finally:
        for name, value in original.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


remote_env, remote_train = load_remote()


def arena(module):
    env = module.BombermanFFAEnv(board_size=9, num_agents=3,
                                wooden_wall_density=0, seed=1)
    env.reset()
    return env


def put_bomb(module, env, r, c, owner, timer=1, radius=3):
    env.bombs.append(module.Bomb(r, c, owner, timer, radius))
    env.agents[owner].bombs_placed += 1


def engine_probes(module):
    env = arena(module)
    for agent, pos in zip(env.agents, [(1, 3), (3, 3), (5, 3)]):
        agent.row, agent.col = pos
    put_bomb(module, env, 3, 1, 0)
    put_bomb(module, env, 3, 5, 2)
    info = env.step([0, 0, 0])[3]
    overlapping = {"kills": info["kills"], "counts": [a.kills for a in env.agents]}

    env = arena(module)
    for agent, pos in zip(env.agents, [(1, 1), (4, 3), (7, 7)]):
        agent.row, agent.col = pos
    put_bomb(module, env, 3, 2, 0, radius=1)
    env.step([0, 0, 0])
    info = env.step([0, module.ACTION_UP, 0])[3]
    lingering = {"victim_alive": env.agents[1].alive, "kills": info["kills"],
                 "counts": [a.kills for a in env.agents]}
    return {"overlapping_blasts": overlapping, "lingering_flame": lingering}


def remote_policy(checkpoint, current_channels=False):
    trainer = local_train if current_channels else remote_train
    net = trainer.ActorCriticNet(board_size=9).to("cpu")
    trainer.smart_load_weights(net, str(checkpoint), "cpu")
    net.eval()
    policy = remote_train.SnapshotOpponent(net, "cpu")
    view = remote_env.BombermanFFAEnv(board_size=9, num_agents=2, seed=0,
                                    sudden_death=False)

    @torch.no_grad()
    def fn(env, obs_list, aid):
        # Read-only view: remote observation and mask, common current engine state.
        for name in ("grid", "agents", "bombs", "flames", "powerups", "step_count",
                     "bomb_timer", "flame_duration", "board_size", "num_agents"):
            setattr(view, name, getattr(env, name))
        obs = view._get_obs(aid)
        assert np.array_equal(obs[:8], obs_list[aid][:8])
        return policy.select_action(obs_list[aid] if current_channels else obs, view, aid)
    return fn


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = Path(__file__).with_name("benchmark_18000.json")
    probes = {"current": engine_probes(local_env), "Loc_update": engine_probes(remote_env)}
    assert probes["current"]["overlapping_blasts"]["counts"] == [1, 0, 0]
    assert probes["Loc_update"]["overlapping_blasts"]["counts"] == [1, 0, 1]
    assert probes["current"]["lingering_flame"]["kills"] == [(0, 1)]
    assert probes["Loc_update"]["lingering_flame"]["kills"] == []

    old_checkpoint = REMOTE / "checkpoints" / "best_phaseC.pt"
    latest_checkpoint = ROOT / "runs" / "pressure-C-next" / "final_phaseC.pt"
    assert sha(old_checkpoint) == sha(ROOT / "checkpoints" / "best_phaseC.pt")
    # Migration must preserve unmasked outputs for the shared eight-channel weights.
    old_net = local_train.ActorCriticNet(board_size=9).eval()
    local_train.smart_load_weights(old_net, str(old_checkpoint), "cpu")
    loc_net = remote_train.ActorCriticNet(board_size=9).eval()
    remote_train.smart_load_weights(loc_net, str(old_checkpoint), "cpu")
    obs = local_env.BombermanFFAEnv(board_size=9, num_agents=2, seed=1).reset()[0]
    x = torch.from_numpy(obs).unsqueeze(0)
    remote_x = torch.cat([x[:, :8], torch.zeros_like(x[:, :1])], dim=1)
    with torch.no_grad():
        for left, right in zip(old_net(x), loc_net(remote_x)):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)

    cases = [
        ("shared_old_checkpoint_current_policy", old_checkpoint, make_ac_fn),
        ("shared_old_checkpoint_Loc_update_policy", old_checkpoint, remote_policy),
        ("current_selected_checkpoint_current_policy", latest_checkpoint, make_ac_fn),
    ]
    data = {"protocol": {"engine": "current working tree", "opponent": "current RuleBot",
                         "board_size": 9, "num_agents": 2, "num_episodes": 100,
                         "max_steps": 500, "seed": 18000, "device": "cpu",
                         "threads": 1, "sudden_death": False,
                         "sampling": "Categorical; same board and policy seed at both corners",
                         "remote_policy_adapter": "remote observation/mask on read-only common engine state"},
            "engine_probes": probes,
            "code_hashes": {str(path.relative_to(ROOT)): sha(path)
                            for folder in (ROOT, REMOTE)
                            for path in (folder / name for name in
                                         ("bomberman_env.py", "rule_based_bot.py", "train_actor_critic.py", "evaluate.py"))},
            "checkpoints": {str(path.relative_to(ROOT)): sha(path)
                            for path in (old_checkpoint, latest_checkpoint)},
            "cases": {}}
    for label, checkpoint, factory in cases:
        print("START", label, flush=True)
        fn = factory(str(checkpoint), 9, "cpu") if factory is make_ac_fn else factory(checkpoint)
        started = time.perf_counter()
        def progress(done, total):
            if done % 20 == 0:
                print(label, done, "/", total, flush=True)
        result = evaluate_seeded(fn, num_episodes=100, seed=18000, max_steps=500,
                                 on_progress=progress)
        result["seconds"] = time.perf_counter() - started
        data["cases"][label] = result
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("RESULT", label, json.dumps({k: v for k, v in result.items()
                                           if k not in ("games", "by_corner")}), flush=True)


if __name__ == "__main__":
    main()
