"""Multi-tier MEC environment for RABS-enabled SAGINs (two-timescale version).

Nodes: RABS (agents), ground MEC servers (GMEC), and LEO satellites.
One environment step = one tactical time slot (1 s). An episode lasts
``max_time_steps`` slots; RABS positions are supplied by the outer-loop
deployment agent through ``reset(rabs_new_positions=...)``.
"""
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import dgl
import networkx as nx
import numpy as np
import torch


class MultiAgentEnv(ABC):
    def seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: Dict[str, int]):
        raise NotImplementedError


# ------------------------------------------------------------------
# Task and RABS state
# ------------------------------------------------------------------
@dataclass
class Task:
    id: str
    data_size: float          # Mbits
    compute_demand: float     # Megacycles
    source_location: Tuple[float, float]
    deadline: float           # seconds
    arrival_time: float
    assigned_rabs: Optional[str] = None
    completed: bool = False
    completion_time: Optional[float] = None


@dataclass
class RABSState:
    id: str
    position: Tuple[float, float]
    battery_level: float
    compute_capacity: float   # MIPS
    queue_backlog: int
    hidden_state: torch.Tensor
    task_queue: deque
    energy_consumed: float
    tasks_processed: int


DEFAULT_CONFIG: Dict[str, Any] = {
    "num_rabs": 10,
    "num_gmec": 4,
    "num_leo": 2,
    "area_size": (1000, 1000),
    "max_time_steps": 60,
    "lambda_task": 2.0,   # Poisson task arrival rate (tasks per slot, system-wide)
    "eta_min": 0.05,      # Minimum link quality to form an edge
    "B_max": 500.0,       # Battery capacity (J)
    "Q_max": 20,          # Queue capacity (tasks)
    "hidden_size": 64,    # Size of the (zero-initialised) per-node hidden state
    "seed": 42,
    "C_min": 1.0,         # Minimum link capacity for action masking (Mbps)
}


class MultiTierMECEnv(MultiAgentEnv):
    # Computation energy per Megacycle (J)
    E_PER_MIPS_RABS = 8e-4
    E_PER_MIPS_GMEC = 2e-4
    E_PER_MIPS_LEO = 4e-4

    LEO_ALTITUDE_M = 550_000.0   # 550 km

    P_T_RABS_DBM = 20.0          # Small cell (ground / air)
    P_T_LEO_DBM = 43.0           # Satellite (~20 W)
    G_RABS_DBI = 4.0             # Omni-directional
    G_LEO_DBI = 30.0             # High-gain beamforming

    BATTERY_DRAIN_SCALE = 0.1    # Fraction of consumed energy drawn from battery

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = dict(DEFAULT_CONFIG)

        self.config = config
        self.num_rabs = config.get("num_rabs")
        self.num_gmec = config.get("num_gmec")
        self.num_leo = config.get("num_leo")
        self.area_size = tuple(config.get("area_size"))
        self.max_time_steps = config.get("max_time_steps")
        self.lambda_task = config.get("lambda_task")
        self.eta_min = config.get("eta_min")
        self.B_max = config.get("B_max")
        self.Q_max = config.get("Q_max")
        self.hidden_size = config.get("hidden_size")

        self.E_per_MIPS_rabs = self.E_PER_MIPS_RABS
        self.E_per_MIPS_gmec = self.E_PER_MIPS_GMEC
        self.E_per_MIPS_leo = self.E_PER_MIPS_LEO

        self.leo_altitude = self.LEO_ALTITUDE_M
        self.P_t_rabs = self.P_T_RABS_DBM
        self.P_t_leo = self.P_T_LEO_DBM
        self.P_t_rabs_linear = 10 ** ((self.P_t_rabs - 30) / 10)   # W
        self.P_t_leo_linear = 10 ** ((self.P_t_leo - 30) / 10)     # W
        self.G_rabs = self.G_RABS_DBI
        self.G_leo = self.G_LEO_DBI

        self.seed(config.get("seed"))

        self.time_step = 0
        self.rabs_agents: Dict[str, RABSState] = {}
        self.gmec_servers: Dict[str, Dict[str, Any]] = {}
        self.leo_satellites: Dict[str, Dict[str, Any]] = {}
        self.network_graph = nx.Graph()
        self.active_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.step_dropped_tasks = 0

        # Positions of RABS/LEO are placeholders until reset()/step().
        self._initialize_entities_structure()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initialize_entities_structure(self):
        # RABS: ~2 GHz CPU (1500-2500 MIPS)
        for i in range(self.num_rabs):
            rabs_id = f"rabs_{i}"
            self.rabs_agents[rabs_id] = RABSState(
                id=rabs_id,
                position=(0.0, 0.0),
                battery_level=self.B_max,
                compute_capacity=np.random.uniform(1500, 2500),
                queue_backlog=0,
                hidden_state=torch.zeros(1, self.hidden_size),
                task_queue=deque(maxlen=self.Q_max),
                energy_consumed=0.0,
                tasks_processed=0,
            )

        # GMEC: fixed positions, 5-10 GHz
        for i in range(self.num_gmec):
            gmec_id = f"gmec_{i}"
            self.gmec_servers[gmec_id] = {
                "id": gmec_id,
                "position": (np.random.uniform(0, self.area_size[0]),
                             np.random.uniform(0, self.area_size[1])),
                "compute_capacity": np.random.uniform(5000, 10000),
                "queue_backlog": 0,
                "energy_consumed": 0.0,
            }

        # LEO: position updated every step by a simple orbit model, ~3 GHz
        for i in range(self.num_leo):
            leo_id = f"leo_{i}"
            self.leo_satellites[leo_id] = {
                "id": leo_id,
                "position": (0.0, 0.0),
                "compute_capacity": np.random.uniform(2500, 4000),
                "queue_backlog": 0,
                "energy_consumed": 0.0,
                "orbital_phase": i * np.pi / max(1, self.num_leo),
            }

    # ------------------------------------------------------------------
    # Physical layer
    # ------------------------------------------------------------------
    def _get_3d_distance(self, pos1, pos2, type1, type2):
        """3D slant range for ground-to-space links, 2D distance otherwise."""
        d_2d = np.linalg.norm(np.array(pos1) - np.array(pos2))
        is_leo_1 = "leo" in type1
        is_leo_2 = "leo" in type2

        if is_leo_1 and is_leo_2:
            # Inter-satellite link (same shell, simplified)
            return max(d_2d, 1.0)
        elif is_leo_1 or is_leo_2:
            # Ground/air-to-space link
            return np.sqrt(d_2d ** 2 + self.leo_altitude ** 2)
        else:
            # Ground/air-to-ground/air link (RABS-GMEC, RABS-RABS)
            return max(d_2d, 1.0)

    def _calculate_link_quality(self, node1_pos, node2_pos, node1_type, node2_type) -> float:
        """Log-distance path loss with log-normal shadowing, mapped to a
        [0, 1] link quality through a sigmoid of the SNR (0.5 at 5 dB)."""
        d = self._get_3d_distance(node1_pos, node2_pos, node1_type, node2_type)

        if "leo" in node1_type or "leo" in node2_type:
            n, sigma = 2.0, 2.0            # space-ground link
            P_t, G_total = self.P_t_leo, self.G_leo
        else:
            n, sigma = 2.2, 1.0            # terrestrial / air LoS link
            P_t, G_total = self.P_t_rabs, self.G_rabs

        path_loss = 30 + 10 * n * np.log10(d) + np.random.normal(0, sigma)
        N0 = -100  # dBm
        snr = P_t + G_total - path_loss - N0

        quality = 1 / (1 + np.exp(-0.4 * (snr - 5.0)))
        return float(np.clip(quality, 0.0, 1.0))

    def _calculate_capacity(self, link_quality: float, distance: float, type1: str, type2: str) -> float:
        """Shannon capacity (Mbps). ``link_quality`` scales the SNR to model
        fading/shadowing attenuation."""
        W = 20e6        # Hz
        f_Hz = 2.4e9    # Hz
        N0_dBm = -100

        if "leo" in type1 or "leo" in type2:
            P_t_dBm, G, n = self.P_t_leo, self.G_leo, 2.0
        else:
            P_t_dBm, G, n = self.P_t_rabs, self.G_rabs, 2.2

        # FSPL constant (f in Hz, d in m) with exponent-scaled distance term
        const_factor = 20 * np.log10(f_Hz) - 147.55
        PL_dB = 10 * n * np.log10(distance + 1e-3) + const_factor

        P_r_dBm = P_t_dBm + G - PL_dB
        SNR_linear = (10 ** ((P_r_dBm - N0_dBm) / 10)) * link_quality
        capacity_bps = W * np.log2(1 + SNR_linear)
        return float(capacity_bps / 1e6)

    def _update_network_topology(self):
        self.network_graph.clear()

        for rid, r in self.rabs_agents.items():
            self.network_graph.add_node(rid, type="rabs", position=r.position)
        for gid, g in self.gmec_servers.items():
            self.network_graph.add_node(gid, type="gmec", position=g["position"])
        for lid, l in self.leo_satellites.items():
            self.network_graph.add_node(lid, type="leo", position=l["position"])

        all_nodes = {**self.rabs_agents, **self.gmec_servers, **self.leo_satellites}
        ids = list(all_nodes.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                node1, node2 = all_nodes[id1], all_nodes[id2]
                pos1 = node1.position if hasattr(node1, "position") else node1["position"]
                pos2 = node2.position if hasattr(node2, "position") else node2["position"]
                type1 = self._node_type(id1)
                type2 = self._node_type(id2)

                lq = self._calculate_link_quality(pos1, pos2, type1, type2)
                if lq > self.eta_min:
                    dist = self._get_3d_distance(pos1, pos2, type1, type2)
                    cap = self._calculate_capacity(lq, dist, type1, type2)
                    self.network_graph.add_edge(id1, id2, link_quality=lq, capacity=cap)

    @staticmethod
    def _node_type(node_id: str) -> str:
        if "rabs" in node_id:
            return "rabs"
        return "gmec" if "gmec" in node_id else "leo"

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------
    def _generate_tasks(self):
        self.step_dropped_tasks = 0
        num_new = np.random.poisson(self.lambda_task)

        for _ in range(num_new):
            task_size_mbits = np.random.uniform(1.0, 5.0)
            task_cpu_cycles = np.random.uniform(100, 500)   # Megacycles

            task = Task(
                id=f"t_{self.time_step}_{len(self.active_tasks)}",
                data_size=task_size_mbits,
                compute_demand=task_cpu_cycles,
                source_location=(np.random.uniform(0, self.area_size[0]),
                                 np.random.uniform(0, self.area_size[1])),
                deadline=np.random.uniform(5.0, 15.0),
                arrival_time=self.time_step,
            )

            # Associate the task with the nearest RABS
            nearest_rabs = min(
                self.rabs_agents.keys(),
                key=lambda r: np.linalg.norm(
                    np.array(self.rabs_agents[r].position) - np.array(task.source_location)
                ),
            )

            if len(self.rabs_agents[nearest_rabs].task_queue) < self.Q_max:
                task.assigned_rabs = nearest_rabs
                self.rabs_agents[nearest_rabs].task_queue.append(task)
                self.rabs_agents[nearest_rabs].queue_backlog += 1
                self.active_tasks.append(task)
            else:
                self.step_dropped_tasks += 1

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_local_features(self, rabs_id: str) -> np.ndarray:
        rabs = self.rabs_agents[rabs_id]
        return np.array([
            rabs.queue_backlog / self.Q_max,
            rabs.battery_level / self.B_max,
            rabs.compute_capacity / 2000.0,
            0.5,   # placeholder (reserved feature)
            0.5,   # placeholder (reserved feature)
        ], dtype=np.float32)

    def _build_agent_dense_observation(self, rabs_id: str) -> np.ndarray:
        """Flat, fixed-size observation vector (for MLP-based agents)."""
        local = self._get_local_features(rabs_id)
        hidden = self.rabs_agents[rabs_id].hidden_state.squeeze(0).numpy()

        neighbors = list(self.network_graph.neighbors(rabs_id))
        neighbor_feats = []
        max_neighbors = self.num_rabs + self.num_gmec + self.num_leo

        for i in range(max_neighbors):
            if i < len(neighbors):
                nid = neighbors[i]
                lq = self.network_graph.get_edge_data(rabs_id, nid)["link_quality"]
                if nid in self.rabs_agents:
                    n_q = self.rabs_agents[nid].queue_backlog / self.Q_max
                    n_c = self.rabs_agents[nid].compute_capacity / 2000.0
                elif nid in self.gmec_servers:
                    n_q = self.gmec_servers[nid]["queue_backlog"] / self.Q_max
                    n_c = self.gmec_servers[nid]["compute_capacity"] / 5000.0
                else:
                    n_q = self.leo_satellites[nid]["queue_backlog"] / self.Q_max
                    n_c = self.leo_satellites[nid]["compute_capacity"] / 3000.0
                neighbor_feats.extend([lq, n_q, n_c])
            else:
                neighbor_feats.extend([0.0, 0.0, 0.0])

        return np.concatenate(
            [local, hidden, np.array(neighbor_feats, dtype=np.float32)]
        ).astype(np.float32)

    def _build_dgl_graph(self):
        """Directed DGL graph for the GNN agent.

        Edge feature ``weight`` = [link_quality, capacity].
        Node feature ``feat``  = [local(5) | hidden(hidden_size) | one-hot type(3)].
        """
        all_nodes = {**self.rabs_agents, **self.gmec_servers, **self.leo_satellites}
        node_ids = sorted(all_nodes.keys())
        node_idx = {nid: i for i, nid in enumerate(node_ids)}

        src, dst, edge_feats = [], [], []
        for id1 in node_ids:
            for id2 in node_ids:
                if id1 == id2:
                    continue
                if self.network_graph.has_edge(id1, id2):
                    edge = self.network_graph.get_edge_data(id1, id2)
                    src.append(node_idx[id1])
                    dst.append(node_idx[id2])
                    edge_feats.append([edge["link_quality"], edge["capacity"]])

        g = dgl.graph((src, dst), num_nodes=len(node_ids))
        if edge_feats:
            g.edata["weight"] = torch.tensor(edge_feats, dtype=torch.float32)
        else:
            g.edata["weight"] = torch.zeros((0, 2), dtype=torch.float32)

        feat_len = 5 + self.hidden_size + 3
        node_feats = []
        for nid in node_ids:
            feat = np.zeros(feat_len, dtype=np.float32)
            if nid in self.rabs_agents:
                feat[0:5] = self._get_local_features(nid)
                feat[5:5 + self.hidden_size] = self.rabs_agents[nid].hidden_state.squeeze(0).numpy()
                feat[-3:] = [1.0, 0.0, 0.0]
            elif nid in self.gmec_servers:
                feat[0] = self.gmec_servers[nid]["queue_backlog"] / self.Q_max
                feat[2] = self.gmec_servers[nid]["compute_capacity"] / 5000.0
                feat[-3:] = [0.0, 1.0, 0.0]
            elif nid in self.leo_satellites:
                feat[0] = self.leo_satellites[nid]["queue_backlog"] / self.Q_max
                feat[2] = self.leo_satellites[nid]["compute_capacity"] / 3000.0
                feat[-3:] = [0.0, 0.0, 1.0]
            node_feats.append(feat)

        g.ndata["feat"] = torch.tensor(np.stack(node_feats), dtype=torch.float32)
        return g, node_idx

    # ------------------------------------------------------------------
    # Actions, masking, reward
    # ------------------------------------------------------------------
    def _get_action_target_nodes_for_agent(self, agent_id: str) -> List[str]:
        """Offloading targets in fixed order: GMEC -> LEO -> RABS.
        Action 0 = local; action k = targets[k - 1]."""
        targets = []
        targets.extend(sorted(self.gmec_servers.keys()))
        targets.extend(sorted(self.leo_satellites.keys()))
        targets.extend(sorted(self.rabs_agents.keys()))
        return targets

    def _calculate_reward_ee(self, rabs_id, action, task, energy_consumed) -> float:
        """Energy-efficiency reward (Mbits/J), clipped at 50."""
        if energy_consumed <= 0 and not task:
            return -5.0
        if task.completed or task.assigned_rabs:
            ee = task.data_size / max(energy_consumed, 1e-9)
            return float(min(ee, 50.0))
        return 0.0

    def get_action_mask(self, rabs_id: str, current_task=None) -> np.ndarray:
        """Binary mask over actions (1 = valid). Local processing (0) is
        always valid. An offloading target is valid if a link exists, its
        capacity is >= C_min, and transmission + compute time <= deadline."""
        rabs = self.rabs_agents[rabs_id]
        targets = self._get_action_target_nodes_for_agent(rabs_id)
        mask = np.zeros(1 + len(targets), dtype=np.int8)
        mask[0] = 1

        if current_task is not None:
            task = current_task
        elif rabs.task_queue:
            task = rabs.task_queue[0]
        else:
            return mask

        C_min = self.config.get("C_min", 1.0)

        for i, target_id in enumerate(targets):
            action_idx = i + 1
            if target_id == rabs_id:
                continue  # no self-offloading
            if not self.network_graph.has_edge(rabs_id, target_id):
                continue

            capacity = self.network_graph.get_edge_data(rabs_id, target_id).get("capacity", 0.0)
            if capacity < C_min:
                continue

            transmission_time = task.data_size / capacity
            if target_id in self.gmec_servers:
                comp_time = task.compute_demand / self.gmec_servers[target_id]["compute_capacity"]
            elif target_id in self.leo_satellites:
                comp_time = task.compute_demand / self.leo_satellites[target_id]["compute_capacity"]
            elif target_id in self.rabs_agents:
                comp_time = task.compute_demand / self.rabs_agents[target_id].compute_capacity
            else:
                comp_time = 0.0

            if transmission_time + comp_time <= task.deadline:
                mask[action_idx] = 1

        return mask

    # ------------------------------------------------------------------
    # Step / reset
    # ------------------------------------------------------------------
    def step(self, actions: Dict[str, int]):
        """Advance one slot. Returns ``obs, (dgl_graph, node_idx), rewards, dones, infos``."""
        self.time_step += 1
        self._generate_tasks()

        # LEO orbit update (projected ground track)
        for leo in self.leo_satellites.values():
            leo["orbital_phase"] += 0.02
            leo["position"] = (
                self.area_size[0] / 2 + 300 * np.cos(leo["orbital_phase"]),
                self.area_size[1] / 2 + 100 * np.sin(leo["orbital_phase"]),
            )

        self._update_network_topology()
        dgl_graph, node_idx = self._build_dgl_graph()

        rewards, dones, infos = {}, {}, {}
        step_total_bits = 0.0
        step_total_energy = 0.0

        for rabs_id in list(self.rabs_agents.keys()):
            rabs = self.rabs_agents[rabs_id]
            action = actions.get(rabs_id, 0)
            step_reward = 0.0

            if rabs.task_queue:
                task = rabs.task_queue.popleft()
                rabs.queue_backlog = max(0, rabs.queue_backlog - 1)
                energy_cost = 0.0
                task_bits = 0.0

                # Enforce constraints C1/C2: invalid choices fall back to local.
                valid_actions_mask = self.get_action_mask(rabs_id, current_task=task)
                if valid_actions_mask[action] == 0:
                    action = 0
                    step_reward -= 0.5

                if action == 0:  # local execution
                    energy_cost = task.compute_demand * self.E_per_MIPS_rabs
                    task.completed = True
                    rabs.tasks_processed += 1
                    task_bits = task.data_size
                    rabs.energy_consumed += energy_cost
                    rabs.battery_level -= energy_cost * self.BATTERY_DRAIN_SCALE

                else:  # offloading
                    targets = self._get_action_target_nodes_for_agent(rabs_id)
                    if (action - 1) < len(targets):
                        target_id = targets[action - 1]

                        if self.network_graph.has_edge(rabs_id, target_id):
                            edge = self.network_graph.get_edge_data(rabs_id, target_id)
                            capacity = max(edge.get("capacity", 1e-6), 1e-6)

                            P_tx_watts = self.P_t_leo_linear if "leo" in target_id else self.P_t_rabs_linear
                            transmission_time = task.data_size / capacity
                            E_tx = P_tx_watts * transmission_time

                            if target_id in self.gmec_servers:
                                E_comp = task.compute_demand * self.E_per_MIPS_gmec
                                self.gmec_servers[target_id]["queue_backlog"] += 1
                                self.gmec_servers[target_id]["energy_consumed"] += E_comp
                            elif target_id in self.leo_satellites:
                                E_comp = task.compute_demand * self.E_per_MIPS_leo
                                self.leo_satellites[target_id]["queue_backlog"] += 1
                                self.leo_satellites[target_id]["energy_consumed"] += E_comp
                            elif target_id in self.rabs_agents:
                                E_comp = task.compute_demand * self.E_per_MIPS_rabs
                                tgt = self.rabs_agents[target_id]
                                if len(tgt.task_queue) < self.Q_max:
                                    tgt.task_queue.append(task)
                                    tgt.queue_backlog += 1
                            else:
                                E_comp = 0.0

                            energy_cost = E_tx + E_comp
                            rabs.energy_consumed += E_tx
                            rabs.battery_level -= E_tx * self.BATTERY_DRAIN_SCALE

                            task.assigned_rabs = target_id
                            task_bits = task.data_size
                        else:
                            step_reward = -5.0   # link failure
                    else:
                        step_reward = -5.0       # invalid target index

                step_total_energy += energy_cost
                step_total_bits += task_bits

                if step_reward == 0.0:
                    step_reward = self._calculate_reward_ee(rabs_id, action, task, energy_cost)
            else:
                step_reward = 0.0  # idle

            if rabs.queue_backlog >= self.Q_max:
                step_reward -= 10.0  # queue overflow penalty

            rewards[rabs_id] = float(step_reward)
            dones[rabs_id] = self.time_step >= self.max_time_steps or rabs.battery_level <= 0

        # Remote servers process one task per slot
        for g in self.gmec_servers.values():
            if g["queue_backlog"] > 0:
                g["queue_backlog"] -= 1
        for l in self.leo_satellites.values():
            if l["queue_backlog"] > 0:
                l["queue_backlog"] -= 1

        global_done = self.time_step >= self.max_time_steps
        for k in dones:
            dones[k] = dones[k] or global_done

        infos["global_network_stats"] = {
            "total_bits": step_total_bits,
            "total_energy": step_total_energy,
        }
        infos["action_masks"] = {rid: self.get_action_mask(rid) for rid in self.rabs_agents}

        obs = {rid: self._build_agent_dense_observation(rid) for rid in self.rabs_agents}
        return obs, (dgl_graph, node_idx), rewards, dones, infos

    def reset(self, rabs_new_positions=None):
        """Reset the episode.

        Args:
            rabs_new_positions: optional array of shape ``[num_rabs, 2]`` in
                ``[-1, 1]`` (deployment-actor output); mapped to the area.
                If ``None``, RABS are placed uniformly at random.

        Note: the RNGs are re-seeded with ``config["seed"]`` on every reset,
        exactly as in the experiments reported in the paper.
        """
        if self.config.get("seed") is not None:
            np.random.seed(self.config.get("seed"))
            torch.manual_seed(self.config.get("seed"))
            random.seed(self.config.get("seed"))

        self.time_step = 0
        self.active_tasks = []

        if rabs_new_positions is not None:
            pos_norm = np.clip(rabs_new_positions, -1.0, 1.0)
            for i, rid in enumerate(sorted(self.rabs_agents.keys())):
                x = (pos_norm[i][0] + 1) / 2 * self.area_size[0]
                y = (pos_norm[i][1] + 1) / 2 * self.area_size[1]
                self.rabs_agents[rid].position = (float(x), float(y))
        else:
            for rid in self.rabs_agents:
                self.rabs_agents[rid].position = (
                    np.random.uniform(0, self.area_size[0]),
                    np.random.uniform(0, self.area_size[1]),
                )

        for r in self.rabs_agents.values():
            r.battery_level = self.B_max
            r.queue_backlog = 0
            r.hidden_state = torch.zeros(1, self.hidden_size)
            r.task_queue.clear()
            r.energy_consumed = 0.0
            r.tasks_processed = 0

        for g in self.gmec_servers.values():
            g["queue_backlog"] = 0
            g["energy_consumed"] = 0.0
        for l in self.leo_satellites.values():
            l["queue_backlog"] = 0
            l["energy_consumed"] = 0.0

        self._update_network_topology()
        dgl_graph, node_idx = self._build_dgl_graph()
        obs = {rid: self._build_agent_dense_observation(rid) for rid in self.rabs_agents}
        return obs, (dgl_graph, node_idx)
