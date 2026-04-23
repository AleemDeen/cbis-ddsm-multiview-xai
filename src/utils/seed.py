import random
import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random seeds across all relevant libraries for reproducible experiments.

    deterministic=True forces cuDNN to use deterministic algorithms, which
    eliminates non-determinism from convolution operations on GPU. benchmark=False
    disables the auto-tuner that would otherwise select different algorithms
    across runs. Together these ensure identical outputs given the same seed,
    at a small cost to training throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass
