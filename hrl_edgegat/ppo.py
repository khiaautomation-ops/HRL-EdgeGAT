"""Masked action selection and PPO update for the offloading agent."""
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_mask(agent_logits, mask_data):
    """Set logits of invalid actions to -1e9 (probability ~0 after softmax)."""
    mask_bool = torch.tensor(mask_data, dtype=torch.bool, device=agent_logits.device)
    if mask_bool.shape[0] != agent_logits.shape[0]:
        mask_bool = mask_bool[:agent_logits.shape[0]]
    agent_logits[~mask_bool] = -1e9
    return agent_logits


def select_actions_gnn(actor, g, node_idx, action_masks=None):
    """Sample one action per RABS from the masked policy.

    Returns:
        actions (dict), log_probs (dict), latency_ms (float): wall-clock time
        of the actor forward pass, i.e. the inference latency in the paper.
    """
    start = time.perf_counter()
    logits = actor(g)
    latency_ms = (time.perf_counter() - start) * 1000

    actions, log_probs = {}, {}
    for aid, idx in node_idx.items():
        if not aid.startswith("rabs"):
            continue
        agent_logits = logits[idx].clone()
        if action_masks and aid in action_masks:
            agent_logits = _apply_mask(agent_logits, action_masks[aid])

        prob = F.softmax(agent_logits, dim=0)
        a = torch.distributions.Categorical(prob).sample().item()

        actions[aid] = int(a)
        log_probs[aid] = torch.log(prob[a] + 1e-10).detach()

    return actions, log_probs, latency_ms


def compute_discounted_returns(rewards_list, gamma=0.99):
    discounted = defaultdict(list)
    if not rewards_list:
        return discounted
    for aid in rewards_list[0].keys():
        G = 0.0
        tmp = []
        for t in reversed(range(len(rewards_list))):
            G = rewards_list[t][aid] + gamma * G
            tmp.insert(0, G)
        discounted[aid] = tmp
    return discounted


def ppo_update_offloading(actor, critic, optimizer, graphs, actions_list, log_probs_list,
                          rewards_list, node_idx_list, masks_list, clip_eps=0.2, epochs=4,
                          entropy_coef=0.03, gamma=0.99, value_coef=0.5, max_grad_norm=0.5):
    """Shared-parameter PPO update over one episode, with batch-wide advantage
    normalisation and the same action masks used at sampling time."""
    discounted = compute_discounted_returns(rewards_list, gamma)
    total_loss = 0

    # Advantage statistics for normalisation
    all_advs = []
    for t, g in enumerate(graphs):
        with torch.no_grad():
            values = critic(g)
            for aid in actions_list[t]:
                if aid not in node_idx_list[t]:
                    continue
                idx = node_idx_list[t][aid]
                ret = torch.tensor(discounted[aid][t], dtype=torch.float32)
                all_advs.append(ret - values[idx])

    if not all_advs:
        return 0.0
    adv_tensor = torch.stack(all_advs)
    adv_mean = adv_tensor.mean()
    adv_std = adv_tensor.std() + 1e-8

    for _ in range(epochs):
        optimizer.zero_grad()
        loss_batch = 0
        count = 0

        for t, g in enumerate(graphs):
            logits = actor(g)
            values = critic(g)
            node_idx = node_idx_list[t]

            step_loss = 0
            step_agents = 0

            for aid, action in actions_list[t].items():
                if aid not in node_idx:
                    continue
                idx = node_idx[aid]
                old_log_prob = log_probs_list[t][aid]
                ret = torch.tensor(discounted[aid][t], dtype=torch.float32)

                agent_logits = logits[idx].clone()
                if masks_list and aid in masks_list[t]:
                    agent_logits = _apply_mask(agent_logits, masks_list[t][aid])

                prob = F.softmax(agent_logits, dim=0)
                log_prob = torch.log(prob[action] + 1e-10)
                entropy = -(prob * torch.log(prob + 1e-10)).sum()

                val = values[idx]
                norm_adv = ((ret - val.detach()) - adv_mean) / adv_std

                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * norm_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_adv

                l_actor = -torch.min(surr1, surr2)
                l_critic = F.mse_loss(val, ret)

                step_loss += l_actor + value_coef * l_critic - entropy_coef * entropy
                step_agents += 1

            if step_agents > 0:
                loss_batch += step_loss / step_agents
                count += 1

        if count > 0:
            final_loss = loss_batch / count
            final_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += final_loss.item()

    return total_loss / epochs
