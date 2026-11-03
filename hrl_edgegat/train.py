"""Two-timescale training loop (Algorithm 1 in the paper).

Outer loop: deployment actor chooses RABS positions once per relocation epoch
(default 3600 tactical slots = 60 episodes of 60 s) and is updated with a
policy gradient on the epoch-level energy efficiency R(k).
Inner loop: shared-parameter EdgeGAT policy chooses offloading targets every
slot and is updated with PPO after every episode.
"""
import csv
import os
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .env import MultiTierMECEnv
from .models import DeploymentActor, DeploymentCritic, GNNActor, GNNCritic
from .ppo import ppo_update_offloading, select_actions_gnn

DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "num_episodes": 3000,
    "gnn_hidden": 128,
    "num_heads": 2,
    "lr_offload": 3e-4,
    "lr_deploy": 1e-3,
    "gamma": 0.99,
    "ppo_epochs": 4,
    "clip_eps": 0.2,
    "relocation_epoch": 3600,   # tactical slots per deployment epoch (T_epoch)
    "decay_rate": 0.997,        # per-episode decay of deployment noise / entropy
    "init_std": 1.0,
    "min_std": 0.01,
    "init_entropy": 0.03,
    "min_entropy": 0.005,
    "offload_reward_divisor": 10.0,
    "deploy_reward_divisor": 10.0,
    "log_every": 10,
}


def _gmec_state(env):
    """Outer-loop state S_k: GMEC coordinates normalised to [-1, 1]."""
    gmec_pos = []
    for i in range(env.num_gmec):
        pos = env.gmec_servers[f"gmec_{i}"]["position"]
        gmec_pos.extend([(pos[0] / env.area_size[0]) * 2 - 1,
                         (pos[1] / env.area_size[1]) * 2 - 1])
    return torch.tensor(gmec_pos, dtype=torch.float32).unsqueeze(0)


def train_two_timescale_ppo(env_config: Dict[str, Any],
                            train_config: Optional[Dict[str, Any]] = None,
                            output_csv: Optional[str] = None,
                            checkpoint_path: Optional[str] = None,
                            verbose: bool = True):
    """Train HRL-EdgeGAT.

    Returns:
        dict with ``ee_history`` (Mbits/J per episode) and
        ``latency_history`` (mean actor inference latency per episode, ms).
    """
    cfg = dict(DEFAULT_TRAIN_CONFIG)
    if train_config:
        cfg.update(train_config)

    env = MultiTierMECEnv(env_config)

    # --- Offloading agent (inner loop) ---
    _, (sample_g, _) = env.reset()
    node_feat_size = sample_g.ndata["feat"].shape[1]
    sample_rabs = list(env.rabs_agents.keys())[0]
    n_offload_actions = 1 + len(env._get_action_target_nodes_for_agent(sample_rabs))

    offload_actor = GNNActor(node_feat_size, cfg["gnn_hidden"], n_offload_actions, cfg["num_heads"])
    offload_critic = GNNCritic(node_feat_size, cfg["gnn_hidden"], cfg["num_heads"])
    offload_optim = optim.Adam(
        list(offload_actor.parameters()) + list(offload_critic.parameters()), lr=cfg["lr_offload"]
    )

    # --- Deployment agent (outer loop) ---
    deploy_input_dim = env.num_gmec * 2
    deploy_actor = DeploymentActor(deploy_input_dim, env.num_rabs)
    deploy_critic = DeploymentCritic(deploy_input_dim)
    deploy_optim = optim.Adam(
        list(deploy_actor.parameters()) + list(deploy_critic.parameters()), lr=cfg["lr_deploy"]
    )

    ee_history, latency_history = [], []
    avg_ee_window = deque(maxlen=100)

    if verbose:
        print("Starting two-timescale PPO (Edge-Aware GAT + annealed deployment)...")

    global_time = 0
    time_since_last_deployment = 0
    relocation_epoch = cfg["relocation_epoch"]

    current_rabs_positions = None
    current_deploy_action = None
    deploy_state = None
    epoch_total_bits = 0.0
    epoch_total_energy = 0.0

    for ep in range(cfg["num_episodes"]):
        # Annealed exploration for deployment noise and offloading entropy
        current_std = max(cfg["min_std"], cfg["init_std"] * (cfg["decay_rate"] ** ep))
        current_entropy = max(cfg["min_entropy"], cfg["init_entropy"] * (cfg["decay_rate"] ** ep))

        # ---------------- Phase A: deployment (start of each epoch) ----------------
        if time_since_last_deployment == 0:
            deploy_state = _gmec_state(env)
            deploy_mu = deploy_actor(deploy_state)
            deploy_action = torch.distributions.Normal(deploy_mu, current_std).sample()
            current_deploy_action = deploy_action.detach()
            current_rabs_positions = (
                torch.clamp(deploy_action, -1, 1).squeeze(0).detach().numpy().reshape(env.num_rabs, 2)
            )
            epoch_total_bits = 0.0
            epoch_total_energy = 0.0

        obs, (g, node_idx) = env.reset(rabs_new_positions=current_rabs_positions)

        # ---------------- Phase B: offloading (T tactical slots) ----------------
        done = {rid: False for rid in node_idx if rid.startswith("rabs")}
        graphs, actions_list, log_probs_list, rewards_list, node_idx_list, masks_list = [], [], [], [], [], []
        ep_total_bits = 0.0
        ep_total_energy = 0.0
        latencies = []

        while not all(done.values()):
            current_masks = {rid: env.get_action_mask(rid) for rid in node_idx if rid.startswith("rabs")}
            actions, log_probs, latency_ms = select_actions_gnn(
                offload_actor, g, node_idx, action_masks=current_masks
            )
            latencies.append(latency_ms)

            _, (next_g, next_node_idx), rewards, done, infos = env.step(actions)

            stats = infos.get("global_network_stats", {"total_bits": 0, "total_energy": 0})
            ep_total_bits += stats["total_bits"]
            ep_total_energy += stats["total_energy"]
            epoch_total_bits += stats["total_bits"]
            epoch_total_energy += stats["total_energy"]

            global_time += 1
            time_since_last_deployment += 1

            scaled_rewards = {k: float(v) / cfg["offload_reward_divisor"] for k, v in rewards.items()}

            graphs.append(g)
            actions_list.append(actions)
            log_probs_list.append(log_probs)
            rewards_list.append(scaled_rewards)
            node_idx_list.append(node_idx)
            masks_list.append(current_masks)

            g, node_idx = next_g, next_node_idx

        # ---------------- Phase C: updates ----------------
        ppo_update_offloading(
            offload_actor, offload_critic, offload_optim,
            graphs, actions_list, log_probs_list, rewards_list, node_idx_list, masks_list,
            clip_eps=cfg["clip_eps"], epochs=cfg["ppo_epochs"],
            entropy_coef=current_entropy, gamma=cfg["gamma"],
        )

        if global_time > 0 and time_since_last_deployment >= relocation_epoch:
            current_mu = deploy_actor(deploy_state)
            dist = torch.distributions.Normal(current_mu, current_std)
            deploy_log_prob = dist.log_prob(current_deploy_action).sum(dim=-1)

            # Epoch-level reward R(k), Eq. (8)
            epoch_raw_ee = epoch_total_bits / (epoch_total_energy + 1e-6)
            deploy_reward = torch.tensor([epoch_raw_ee / cfg["deploy_reward_divisor"]], dtype=torch.float32)

            deploy_val = deploy_critic(deploy_state).squeeze(-1)
            deploy_adv = deploy_reward - deploy_val.detach()
            loss_critic = F.mse_loss(deploy_val, deploy_reward)
            loss_actor = -(deploy_log_prob * deploy_adv).mean()

            deploy_optim.zero_grad()
            (loss_actor + 0.5 * loss_critic).backward()
            nn.utils.clip_grad_norm_(deploy_actor.parameters(), 1.0)
            nn.utils.clip_grad_norm_(deploy_critic.parameters(), 1.0)
            deploy_optim.step()

            if verbose:
                print(f">>> DEPLOYMENT UPDATED at {global_time}s | Epoch EE: {epoch_raw_ee:.2f}")
            time_since_last_deployment = 0

        # ---------------- Logging ----------------
        raw_ee = ep_total_bits / (ep_total_energy + 1e-6)
        ee_history.append(raw_ee)
        avg_ee_window.append(raw_ee)
        latency_history.append(np.mean(latencies) if latencies else 0.0)

        if verbose and (ep + 1) % cfg["log_every"] == 0:
            avg_ee = sum(avg_ee_window) / len(avg_ee_window)
            print(f"Ep {ep + 1}: Avg EE={avg_ee:.2f} | Raw EE={raw_ee:.2f} | "
                  f"NoiseStd={current_std:.3f} | AvgLatency={np.mean(latencies):.3f}")

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        # Row 1: EE per episode (Mbits/J). Row 2: mean inference latency (ms).
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ee_history)
            writer.writerow(latency_history)

    if checkpoint_path:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        torch.save({
            "offload_actor": offload_actor.state_dict(),
            "offload_critic": offload_critic.state_dict(),
            "deploy_actor": deploy_actor.state_dict(),
            "deploy_critic": deploy_critic.state_dict(),
            "env_config": env_config,
            "train_config": cfg,
        }, checkpoint_path)

    return {"ee_history": ee_history, "latency_history": latency_history}
