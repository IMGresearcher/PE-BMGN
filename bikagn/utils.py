
from __future__ import annotations

import os
import random
import re
from typing import List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def natural_key(path: str):
    """Natural sort key for file names such as 1.csv, 2.csv, ..., 10.csv."""
    base = os.path.basename(path)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", base)]


def build_rul_labels(num_steps: int) -> torch.Tensor:
    """Create normalized RUL labels from 1 to 0 for a complete run-to-failure sequence."""
    if num_steps == 1:
        return torch.tensor([0.0], dtype=torch.float32)
    vals = [(num_steps - 1 - t) / (num_steps - 1) for t in range(num_steps)]
    return torch.tensor(vals, dtype=torch.float32)


def zscore_fit_from_runs(train_runs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit channel-wise z-score statistics from training bearing runs.

    Each run is expected to have shape [T, C, L].
    """
    all_x = torch.cat(train_runs, dim=0)
    mean = all_x.mean(dim=(0, 2), keepdim=True)
    std = all_x.std(dim=(0, 2), keepdim=True) + 1e-8
    return mean, std


def zscore_apply(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    return float(np.mean(np.abs(pred - target)))


def eas(pred: np.ndarray, target: np.ndarray) -> float:
    """Engineering Application Score used in the paper.

    Smaller values are better. The formula uses exp(error/scale) - 1.
    """
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    scores = []
    for yhat, y in zip(pred, target):
        if y > yhat:
            scores.append(np.exp((y - yhat) / 13.0) - 1.0)
        else:
            scores.append(np.exp((yhat - y) / 10.0) - 1.0)
    return float(np.mean(scores))
