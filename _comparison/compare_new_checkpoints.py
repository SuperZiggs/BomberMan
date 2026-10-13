"""Compare supplied trained checkpoints without modifying either game pipeline."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import compare_branches as c

OUTPUT = c.ROOT / "_comparison" / "updated_results"
SEED = 19000
CHECKPOINTS = {
    "current": c.ROOT / "runs/pressure-C-next/final_phaseC.pt",
    "v18": c.REMOTE / "checkpoints_v18/best_phaseC.pt",
    "v19": c.REMOTE / "checkpoints_v19/final_phaseC.pt",
}
CASES = {
    "current_current_mask": ("current", "current"),
    "v18_current_mask": ("v18", "current"),
    "v19_current_mask": ("v19", "current"),
    "v18_Loc_mask": ("v18", "Loc"),
    "v19_Loc_mask": ("v19", "Loc"),
}


def load_net(checkpoint, current=False):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = payload.get("model_state_dict", payload)
    trainer = c.local_train if current else c.remote_train
    net = trainer.ActorCriticNet(board_size=9).eval()
    assert set(weights) == set(net.state_dict())
    assert all(v.shape == net.state_dict()[k].shape for k, v in weights.items())
    assert all(torch.isfinite(v).all() for v in weights.values())
    net.load_state_dict(weights, strict=True)
    return net


def common_policy(kind, mask_kind):
    net = load_net(CHECKPOINTS[kind], current=kind == "current")
    view = c.remote_env.BombermanFFAEnv(board_size=9, num_agents=2, seed=0,
                                      sudden_death=False)
    @torch.no_grad()
    def fn(env, observations, aid):
        for name in ("grid", "agents", "bombs", "flames", "powerups", "step_count",
                     "bomb_timer", "flame_duration", "board_size", "num_agents"):
            setattr(view, name, getattr(env, name))
        loc_obs = view._get_obs(aid)
        assert np.array_equal(loc_obs[:8], observations[aid][:8])
        assert not loc_obs[8].any()
        obs = observations[aid] if kind == "current" else loc_obs
        logits, _ = net(torch.from_numpy(obs).unsqueeze(0))
        mask = (c.local_train.get_action_mask(env, aid) if mask_kind == "current"
                else c.remote_train.get_action_mask(view, aid))
        assert any(mask)
        logits = logits.masked_fill(~torch.tensor(mask, dtype=torch.bool), -1e9)
        return torch.distributions.Categorical(logits=logits).sample().item()
    return fn


class ObservedLocEnv(c.remote_env.BombermanFFAEnv):
    """Observe real blast sources; keep native movement, rewards and observations."""
    def reset(self):
        self.audit_sources = {}
        self.audit_kill_counts = [0] * self.num_agents
        self.blast_events = []
        return super().reset()

    def _explode_bomb(self, bomb):
        self.blast_events.append((bomb.owner, self._bomb_blast_cells(
            bomb.row, bomb.col, bomb.blast_range), bool(bomb.chain_triggered)))
        return super()._explode_bomb(bomb)

    def step(self, actions):
        alive_before = [agent.alive for agent in self.agents]
        self.blast_events = []
        obs, rewards, dones, info = super().step(actions)
        for owner, cells, chained in self.blast_events:
            for pos in cells:
                sources = self.audit_sources.setdefault(pos, {})
                sources[owner] = (self.flame_duration, chained)
        kills, suicides, chain_self, other = [], [], [], []
        for agent in self.agents:
            aid = agent.agent_id
            if not alive_before[aid] or agent.alive or aid in info["sudden_death_kills"]:
                continue
            sources = self.audit_sources.get((agent.row, agent.col), {})
            if aid in sources:
                suicides.append(aid)
                if sources[aid][1]:
                    chain_self.append(aid)
            owner = next((owner for owner in sources if owner != aid),
                         aid if aid in sources else None)
            if owner is None:
                other.append(aid)
            else:
                kills.append((owner, aid))
                if owner != aid:
                    self.audit_kill_counts[owner] += 1
        for pos in list(self.audit_sources):
            remaining = {owner: (duration - 1, chained)
                         for owner, (duration, chained) in self.audit_sources[pos].items()
                         if duration > 1}
            if remaining and pos in self.flames:
                self.audit_sources[pos] = remaining
            else:
                del self.audit_sources[pos]
        info["audit"] = {"kills": kills, "suicides": suicides,
                         "chain_self": chain_self, "other": other}
        return obs, rewards, dones, info


def check_observer():
    for delayed in (False, True):
        env = ObservedLocEnv(board_size=9, num_agents=3, wooden_wall_density=0, seed=1)
        env.reset()
        for agent, pos in zip(env.agents, [(1, 1), (4, 3) if delayed else (3, 3), (7, 7)]):
            agent.row, agent.col = pos
        c.put_bomb(c.remote_env, env, 3, 2, 0, radius=1)
        info = env.step([0, 0, 0])[3]
        if delayed:
            assert info["audit"]["kills"] == []
            info = env.step([0, c.remote_env.ACTION_UP, 0])[3]
        assert info["audit"]["kills"] == [(0, 1)]
        assert env.audit_kill_counts == [1, 0, 0]
    env = ObservedLocEnv(board_size=9, num_agents=3, wooden_wall_density=0, seed=1)
    env.reset()
    for agent, pos in zip(env.agents, [(1, 3), (3, 3), (5, 3)]):
        agent.row, agent.col = pos
    c.put_bomb(c.remote_env, env, 3, 1, 0)
    c.put_bomb(c.remote_env, env, 3, 5, 2)
    assert env.step([0, 0, 0])[3]["audit"]["kills"] == [(0, 1)]
    assert env.audit_kill_counts == [1, 0, 0]
    env = ObservedLocEnv(board_size=9, num_agents=2, wooden_wall_density=0, seed=1)
    env.reset()
    for aid, pos in enumerate(((1, 1), (1, 3))):
        env.agents[aid].row, env.agents[aid].col = pos
        c.put_bomb(c.remote_env, env, *pos, aid, radius=4)
    audit = env.step([0, 0])[3]["audit"]
    assert audit["suicides"] == [0, 1]
    assert audit["kills"] == [(1, 0), (0, 1)]
    # Verify that observing events leaves the actual simulation unchanged.
    native = c.remote_env.BombermanFFAEnv(board_size=9, num_agents=2, seed=4,
                                        sudden_death=True, sudden_death_start=20)
    observed = ObservedLocEnv(board_size=9, num_agents=2, seed=4,
                             sudden_death=True, sudden_death_start=20)
    native.reset()
    observed.reset()
    actions_rng = np.random.default_rng(9)
    for _ in range(45):
        actions = actions_rng.integers(0, 6, size=2).tolist()
        left, right = native.step(actions), observed.step(actions)
        assert np.array_equal(native.grid, observed.grid)
        assert native.agents == observed.agents and native.bombs == observed.bombs
        assert native.flames == observed.flames and native.powerups == observed.powerups
        assert all(np.array_equal(a, b) for a, b in zip(left[0], right[0]))
        assert left[1:3] == right[1:3]


def summarize(games):
    n = len(games)
    rates = {f"{key}_rate": sum(g[key] for g in games) / n * 100
             for key in ("win", "active_win", "suicide", "draw")}
    averages = {f"avg_{key}": sum(g[key] for g in games) / n
                for key in ("kills", "bombs", "steps")}
    return {"num_episodes": n, **rates, **averages,
            "sd_death_rate": sum(g.get("sd_death", 0) for g in games) / n * 100,
            "chain_self_rate": sum(g.get("chain_self", 0) for g in games) / n * 100,
            "by_corner": {str(aid): {f"{key}_rate": sum(g[key] for g in games if g["aid"] == aid) / (n // 2) * 100
                                    for key in ("win", "active_win", "suicide", "draw")}
                          for aid in range(2)}, "games": games}


def native_evaluation(kind, sudden_death, episodes=100):
    check_observer()
    net = load_net(CHECKPOINTS[kind])
    policy = c.remote_train.SnapshotOpponent(net, "cpu")
    games = []
    with torch.no_grad():
        for ep in range(episodes):
            seed, aid = SEED + ep // 2, ep % 2
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = ObservedLocEnv(board_size=9, num_agents=2, max_steps=300,
                                 sudden_death=sudden_death, seed=seed)
            obs = env.reset()
            start_board = hashlib.sha256(env.grid.tobytes()).hexdigest()
            bot = c.remote_train.RuleBasedAgent(1 - aid)
            bombs, suicides, chain_self, sd = 0, 0, 0, 0
            for tick in range(env.max_steps):
                actions = [0, 0]
                if env.agents[aid].alive:
                    actions[aid] = policy.select_action(obs[aid], env, aid)
                if env.agents[1 - aid].alive:
                    actions[1 - aid] = bot.select_action(env)
                obs, _, _, info = env.step(actions)
                # These native placement entries are emitted only for bombs actually placed.
                bombs += sum(owner == aid for owner, _ in info["bomb_pressure_scores"])
                suicides += aid in info["audit"]["suicides"]
                chain_self += aid in info["audit"]["chain_self"]
                sd += aid in info["sudden_death_kills"]
                if info["episode_done"]:
                    break
            winner = env.agents[aid].alive and sum(info["alive"]) == 1
            games.append({"seed": seed, "aid": aid, "start_board_sha256": start_board,
                          "win": int(winner), "active_win": int(winner and env.audit_kill_counts[aid] > 0),
                          "suicide": int(bool(suicides)), "chain_self": int(bool(chain_self)),
                          "sd_death": int(bool(sd)), "draw": int(sum(info["alive"]) != 1),
                          "kills": env.audit_kill_counts[aid], "raw_native_kills": env.agents[aid].kills,
                          "bombs": bombs, "steps": info["step"]})
            if (ep + 1) % 20 == 0:
                print("NATIVE", kind, "bo", sudden_death, ep + 1, "/", episodes, flush=True)
    assert all(games[i]["start_board_sha256"] == games[i + 1]["start_board_sha256"]
               for i in range(0, episodes, 2))
    return summarize(games)


def run_case(label):
    started = time.perf_counter()
    print("START", label, flush=True)
    if label in CASES:
        kind, mask = CASES[label]
        result = c.evaluate_seeded(common_policy(kind, mask), num_episodes=200, seed=SEED,
                                  max_steps=500, on_progress=lambda n, total:
                                  print(label, n, "/", total, flush=True) if n % 20 == 0 else None)
        protocol = {"engine": "current", "opponent": "current RuleBot", "max_steps": 500,
                    "mask": mask, "sudden_death": False, "episodes": 200}
    else:
        kind, bo = label.split("_")[0], label.endswith("on")
        result = native_evaluation(kind, bo)
        protocol = {"engine": "Loc_update", "opponent": "Loc_update RuleBot", "max_steps": 300,
                    "mask": "Loc", "sudden_death": bo, "episodes": 100,
                    "kill_audit": "observe real blast sources, one credit per victim including lingering flames"}
    result["seconds"] = time.perf_counter() - started
    assert len(result["games"]) == protocol["episodes"]
    assert [(g["seed"], g["aid"]) for g in result["games"]] == [
        (SEED + i // 2, i % 2) for i in range(protocol["episodes"])]
    data = {"case": label, "protocol": {"seed": SEED, "board_size": 9, "num_agents": 2,
            "device": "cpu", "threads": 1, **protocol},
            "checkpoint": str(CHECKPOINTS[kind].relative_to(c.ROOT)),
            "checkpoint_sha256": c.sha(CHECKPOINTS[kind]),
            "code_hashes": {str((folder / name).relative_to(c.ROOT)): c.sha(folder / name)
                            for folder in (c.ROOT, c.REMOTE)
                            for name in ("bomberman_env.py", "rule_based_bot.py", "train_actor_critic.py", "evaluate.py")},
            "result": result}
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / f"{label}_19000.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("RESULT", label, json.dumps({k: v for k, v in result.items()
                                      if k not in ("games", "by_corner")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+", choices=[*CASES, "v18_native_on", "v19_native_on", "v19_native_off"])
    args = parser.parse_args()
    for label in args.cases:
        run_case(label)
