"""Assert the integration risks reproduced during the branch comparison."""
import json
from unittest.mock import patch
import torch
import compare_branches as c


def main():
    old = torch.load(c.REMOTE / "checkpoints" / "best_phaseC.pt",
                     weights_only=True, map_location="cpu")
    net = c.local_train.ActorCriticNet(board_size=9)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    payload = c.remote_train.checkpoint_payload(old, episode=1, phase="C",
                                               metrics={}, config={})
    with patch.object(c.local_train.torch, "load", return_value=payload):
        c.local_train.smart_load_weights(net, "payload.pt", "cpu")
    ignored = all(torch.equal(v, net.state_dict()[k]) for k, v in before.items())

    loc_net = c.remote_train.ActorCriticNet(board_size=9)
    conv = loc_net.conv[0].weight.detach().clone()
    c.remote_train.smart_load_weights(
        loc_net, str(c.ROOT / "runs" / "pressure-C-next" / "final_phaseC.pt"), "cpu")
    dropped = torch.equal(conv, loc_net.conv[0].weight)
    assert ignored and dropped

    local = c.arena(c.local_env)
    remote = c.arena(c.remote_env)
    for env, module in ((local, c.local_env), (remote, c.remote_env)):
        env.grid[3, 3] = module.WOODEN_WALL
        c.put_bomb(module, env, 3, 1, 0, timer=1, radius=2)
        c.put_bomb(module, env, 3, 5, 1, timer=4, radius=4)
    hazards = c.remote_train._project_bomb_hazards(remote)
    burns, _ = local._forecast_blasts()
    assert burns[3, 3, 2] and 4 not in hazards[(3, 2)]

    probes = {"current": c.engine_probes(c.local_env),
              "Loc_update": c.engine_probes(c.remote_env)}
    assert probes["current"]["overlapping_blasts"]["counts"] == [1, 0, 0]
    assert probes["Loc_update"]["overlapping_blasts"]["counts"] == [1, 0, 1]
    assert probes["current"]["lingering_flame"]["kills"] == [(0, 1)]
    assert probes["Loc_update"]["lingering_flame"]["kills"] == []

    result = {
        "current_loader_silently_ignores_metadata_payload": ignored,
        "remote_loader_keeps_random_conv_when_loading_current_12_channel_checkpoint": dropped,
        "engine_probes": probes,
        "static_wall_forecast": {
            "case": "early bomb clears wood at (3,3); later bomb at (3,5) explodes at tick 4",
            "current_forecast_damage_ticks_at_3_2": [i + 1 for i, value in enumerate(burns[:, 3, 2]) if value],
            "Loc_update_forecast_damage_ticks_at_3_2": sorted(hazards[(3, 2)]),
        },
    }
    path = c.ROOT / "_comparison" / "compatibility_checks.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
