#!/usr/bin/env python3
"""Train HRL-EdgeGAT.

Examples:
    python scripts/train.py --config configs/paper_n10.json
    python scripts/train.py --config configs/paper_n32.json
    python scripts/train.py --config configs/paper_n10.json --num-rabs 24 --episodes 500
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hrl_edgegat.utils import limit_cpu_threads, set_seeds  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/paper_n10.json", help="JSON config file")
    p.add_argument("--episodes", type=int, help="override train.num_episodes")
    p.add_argument("--num-rabs", type=int, help="override env.num_rabs")
    p.add_argument("--lambda-task", type=float, help="override env.lambda_task")
    p.add_argument("--seed", type=int, help="override env.seed (also used for global seeding)")
    p.add_argument("--threads", type=int, default=2, help="CPU threads (paper latency measured with 2)")
    p.add_argument("--out-dir", default="results", help="directory for CSV / checkpoint / plot")
    p.add_argument("--tag", default=None, help="run name (default: derived from config)")
    p.add_argument("--no-plot", action="store_true", help="skip saving the convergence plot")
    return p.parse_args()


def main():
    args = parse_args()
    limit_cpu_threads(args.threads)

    with open(args.config) as f:
        cfg = json.load(f)
    env_cfg, train_cfg = cfg["env"], cfg.get("train", {})

    if args.episodes is not None:
        train_cfg["num_episodes"] = args.episodes
    if args.num_rabs is not None:
        env_cfg["num_rabs"] = args.num_rabs
    if args.lambda_task is not None:
        env_cfg["lambda_task"] = args.lambda_task
    if args.seed is not None:
        env_cfg["seed"] = args.seed

    set_seeds(env_cfg.get("seed", 42))

    from hrl_edgegat.train import train_two_timescale_ppo

    tag = args.tag or (f"edgegat_{env_cfg['num_rabs']}_{env_cfg['num_gmec']}_{env_cfg['num_leo']}"
                       f"_lam{env_cfg['lambda_task']:g}_seed{env_cfg['seed']}")
    csv_path = os.path.join(args.out_dir, f"{tag}.csv")
    ckpt_path = os.path.join(args.out_dir, f"{tag}.pt")

    out = train_two_timescale_ppo(env_cfg, train_cfg, output_csv=csv_path, checkpoint_path=ckpt_path)

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(out["ee_history"])
        plt.title("HRL-EdgeGAT training convergence")
        plt.xlabel("Episode")
        plt.ylabel("Energy efficiency (Mbits/J)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{tag}.png"), dpi=150)

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
