"""Neural networks for HRL-EdgeGAT.

Outer loop: MLP deployment actor/critic (continuous RABS positions).
Inner loop: Edge-aware GNN actor/critic (discrete offloading targets).
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


# ==========================================
# Outer loop: deployment agent (MLP)
# ==========================================
class DeploymentActor(nn.Module):
    """Maps GMEC coordinates to mean RABS positions in [-1, 1]."""

    def __init__(self, input_dim, num_rabs, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_size, num_rabs * 2)

    def forward(self, x):
        return torch.tanh(self.mu_head(self.net(x)))


class DeploymentCritic(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# Inner loop: offloading agent (edge-aware GNN)
# ==========================================
class EdgeAwareGAT(nn.Module):
    """Edge-aware aggregation layer: h_i = ReLU( sum_j w_ij * W h_j ),
    where w_ij is the link quality (edge feature index 0)."""

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = nn.ReLU()

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = self.linear(h)
            g.edata["w"] = g.edata["weight"][:, 0].unsqueeze(-1)
            g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h_new"))
            return self.activation(g.ndata["h_new"])


class GNNActor(nn.Module):
    """EdgeAwareGAT -> 2-head GATConv -> per-node action logits."""

    def __init__(self, node_feat_size, hidden_size, n_actions, num_heads=2):
        super().__init__()
        self.gnn1 = EdgeAwareGAT(node_feat_size, hidden_size)
        self.gnn2 = GATConv(hidden_size, hidden_size, num_heads=num_heads, activation=F.relu)
        self.policy = nn.Linear(hidden_size * num_heads, n_actions)

    def forward(self, g):
        h = self.gnn1(g, g.ndata["feat"])
        h = self.gnn2(g, h).flatten(1)
        return self.policy(h)


class GNNCritic(nn.Module):
    """Centralised critic over the global graph (CTDE)."""

    def __init__(self, node_feat_size, hidden_size=128, num_heads=2):
        super().__init__()
        self.gnn1 = EdgeAwareGAT(node_feat_size, hidden_size)
        self.gnn2 = GATConv(hidden_size, hidden_size, num_heads=num_heads, activation=F.relu)
        self.value = nn.Linear(hidden_size * num_heads, 1)

    def forward(self, g):
        h = self.gnn1(g, g.ndata["feat"])
        h = self.gnn2(g, h).flatten(1)
        return self.value(h).squeeze(-1)
