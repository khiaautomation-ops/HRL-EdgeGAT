#!/usr/bin/env python3
"""Plot one or more result CSVs produced by scripts/train.py.

Each CSV has two rows: per-episode energy efficiency (Mbits/J) and
per-episode mean inference latency (ms).

    python scripts/plot_results.py results/edgegat_10_4_2_lam2_seed42.csv --labels EdgeGAT
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return np.array(rows[0], dtype=float), np.array(rows[1], dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--labels", nargs="*")
    p.add_argument("--window", type=int, default=100, help="steady-state window (last N episodes)")
    p.add_argument("--smooth", type=int, default=50, help="moving-average window for the curve")
    p.add_argument("--out", default="results/convergence.png")
    a = p.parse_args()
    labels = a.labels or [os.path.splitext(os.path.basename(c))[0] for c in a.csvs]

    plt.figure(figsize=(6, 4))
    for path, label in zip(a.csvs, labels):
        ee, lat = load(path)
        k = max(1, min(a.smooth, len(ee)))
        curve = np.convolve(ee, np.ones(k) / k, mode="valid")
        plt.plot(np.arange(k - 1, len(ee)), curve, label=label)
        w = min(a.window, len(ee))
        print(f"{label}: steady-state EE = {ee[-w:].mean():.2f} Mbits/J, "
              f"latency = {lat[-w:].mean():.2f} ms (last {w} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Avg energy efficiency (Mbits/J)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    plt.savefig(a.out, dpi=150)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
