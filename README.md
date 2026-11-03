# HRL-EdgeGAT

Official implementation of

> E. Theingi, L. Sboui, and D. Naboulsi, **"Hierarchical Reinforcement Learning with Edge-Aware GAT for Energy-Efficient MEC in RABS-Enabled SAGINs,"** *IEEE Wireless Communications Letters*, 2026. DOI: [10.1109/LWC.2026.3748312](https://doi.org/10.1109/LWC.2026.3748312)

HRL-EdgeGAT is a two-timescale hierarchical reinforcement learning framework for energy-efficient multi-tier MEC load balancing in Space-Air-Ground Integrated Networks (SAGINs) served by Robotic Airborne Base Stations (RABSs). An outer-loop deployment agent places the RABSs once per relocation epoch, and an inner-loop multi-agent policy built on an edge-aware graph attention network (EdgeGAT) chooses, every time slot, whether each RABS processes its task locally or offloads it to a peer RABS, a ground MEC server, or a LEO satellite.

## Repository structure

```
hrl_edgegat/
  env.py       Multi-tier MEC environment (RABS / GMEC / LEO, channel, energy, masking)
  models.py    Deployment actor/critic (MLP) and EdgeGAT actor/critic (DGL)
  ppo.py       Masked action sampling and PPO update for the offloading policy
  train.py     Two-timescale training loop (Algorithm 1)
  utils.py     Seeding and CPU-thread helpers
scripts/
  train.py         Command-line training entry point
  plot_results.py  Convergence plot and steady-state statistics from result CSVs
configs/
  paper_n10.json   Main setting (Fig. 2): 10 RABS, 4 GMEC, 2 LEO
  paper_n32.json   Density study (Fig. 3b): 32 RABS
notebooks/
  colab_quickstart.ipynb   Run everything on Google Colab
tests/
  test_smoke.py    Short end-to-end check
```

## Installation

The experiments were run with Python 3.12 on a Google Colab CPU runtime.

```bash
git clone https://github.com/<your-username>/HRL-EdgeGAT.git
cd HRL-EdgeGAT

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
pip install torchdata==0.11.0 numpy networkx matplotlib
```

DGL wheels are built against specific PyTorch versions, so install the DGL wheel that matches your PyTorch build (see the [DGL install page](https://www.dgl.ai/pages/start.html)).

## Usage

Train with the main configuration (10 RABS, 3000 episodes):

```bash
python scripts/train.py --config configs/paper_n10.json
```

Density study with 32 RABS:

```bash
python scripts/train.py --config configs/paper_n32.json
```

Any setting can be overridden from the command line, for example a traffic-load or density sweep:

```bash
python scripts/train.py --config configs/paper_n10.json --lambda-task 4.0
python scripts/train.py --config configs/paper_n10.json --num-rabs 24
```

Each run writes to `results/`:

| File | Contents |
|---|---|
| `<tag>.csv` | Row 1: energy efficiency (Mbits/J) per episode. Row 2: mean actor inference latency (ms) per episode. |
| `<tag>.pt` | Trained actor/critic weights and the configs used. |
| `<tag>.png` | Training convergence curve. |

Steady-state numbers in the paper are averaged over the last 100 episodes:

```bash
python scripts/plot_results.py results/edgegat_10_4_2_lam2_seed42.csv --labels EdgeGAT
```

The inference latency is the wall-clock time of one actor forward pass over the whole graph, measured with 2 CPU threads (`--threads 2`, the default). Absolute latency values depend on hardware.

A quick check that everything runs (about a minute on CPU):

```bash
python tests/test_smoke.py
```

## Reproducibility

All random number generators are seeded from `env.seed` (default 42). The environment re-seeds its RNGs on every `reset()`, so with the same seed and the same library versions a run is deterministic on CPU. Results can still differ slightly across PyTorch/DGL versions or hardware.

## Implementation notes

These describe how the paper's model is realized in the code:

- One environment step is one 1-s tactical slot and an episode lasts 60 slots. With `relocation_epoch = 3600`, the deployment agent chooses new RABS positions every 60 episodes and is updated with the epoch-level energy efficiency, Eq. (8).
- `lambda_task` is the Poisson arrival rate of new tasks per slot for the whole network. Each task is associated with the nearest RABS.
- Task sizes are 1–5 Mbits and compute demands 100–500 Megacycles. Computation energy is modeled per Megacycle for each tier (`E_PER_MIPS_*` in `env.py`), and transmission energy is transmit power × D / C.
- The edge feature vector is `[link_quality, capacity]`, where link quality is a sigmoid mapping of the instantaneous SNR. The EdgeAwareGAT layer weights neighbor messages by the link quality.
- Action 0 is local processing; actions 1..M index GMEC, LEO, then RABS targets in sorted order. The action mask removes self-offloading, missing links, links below `C_min`, and targets that would miss the task deadline.

## Citation

```bibtex
@article{theingi2026hrledgegat,
  author  = {Theingi, Ei and Sboui, Lokman and Naboulsi, Diala},
  title   = {Hierarchical Reinforcement Learning with Edge-Aware {GAT} for Energy-Efficient {MEC} in {RABS}-Enabled {SAGINs}},
  journal = {IEEE Wireless Communications Letters},
  year    = {2026},
  doi     = {10.1109/LWC.2026.3748312}
}
```

## License

Released under the [MIT License](LICENSE).
