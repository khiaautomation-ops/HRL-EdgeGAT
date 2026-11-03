"""Short end-to-end check: environment, masking, and a few training episodes."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from hrl_edgegat.env import MultiTierMECEnv  # noqa: E402
from hrl_edgegat.train import train_two_timescale_ppo  # noqa: E402
from hrl_edgegat.utils import set_seeds  # noqa: E402

ENV_CFG = {
    "num_rabs": 4, "num_gmec": 2, "num_leo": 1, "area_size": [1000, 1000],
    "max_time_steps": 10, "lambda_task": 2.0, "B_max": 500.0, "Q_max": 20,
    "hidden_size": 16, "seed": 0, "eta_min": 0.05,
}


def test_env_and_mask():
    env = MultiTierMECEnv(dict(ENV_CFG))
    _, (g, node_idx) = env.reset()
    assert g.num_nodes() == 4 + 2 + 1
    for rid in env.rabs_agents:
        mask = env.get_action_mask(rid)
        assert mask[0] == 1
        targets = env._get_action_target_nodes_for_agent(rid)
        assert mask[1 + targets.index(rid)] == 0  # no self-offloading
    _, _, rewards, dones, infos = env.step({rid: 0 for rid in env.rabs_agents})
    assert set(rewards) == set(env.rabs_agents)
    assert "global_network_stats" in infos


def test_training_runs():
    set_seeds(0)
    out = train_two_timescale_ppo(
        dict(ENV_CFG),
        {"num_episodes": 3, "relocation_epoch": 20, "gnn_hidden": 16},
        verbose=False,
    )
    assert len(out["ee_history"]) == 3
    assert all(np.isfinite(out["ee_history"]))


if __name__ == "__main__":
    test_env_and_mask()
    test_training_runs()
    print("Smoke tests passed.")
