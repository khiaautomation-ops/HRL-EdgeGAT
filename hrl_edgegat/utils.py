"""Reproducibility and runtime helpers."""
import os
import random

import numpy as np
import torch


def set_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy, PyTorch (CPU/CUDA) and DGL."""
    import dgl

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def limit_cpu_threads(num_threads: int) -> None:
    """Pin BLAS/OpenMP and PyTorch thread pools.

    The inference latencies reported in the paper were measured with 2 CPU
    threads on a Google Colab CPU runtime.
    """
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_threads)
    except RuntimeError:
        # Can only be set once, before any inter-op parallel work starts.
        pass
