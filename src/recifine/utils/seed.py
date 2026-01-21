from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int, n_gpu: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
