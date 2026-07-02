
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_rul_comparison_plot(
    pred: np.ndarray,
    target: np.ndarray,
    bearing_name: str,
    save_path: str,
    title_prefix: str = None,
) -> None:
    """Save an RUL curve with prediction error displayed below zero."""
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    steps = np.arange(len(target))

    err = pred - target
    err_below = -np.abs(err)

    y_top = max(1.02, float(np.max(target)), float(np.max(pred)))
    y_bottom = min(-0.25, float(np.min(err_below)) * 1.15)

    title = f"{title_prefix} | {bearing_name}" if title_prefix is not None else bearing_name

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10.5, 4.8))
    plt.plot(steps, target, color="black", linewidth=2.6, label="Ground Truth RUL")
    plt.plot(steps, pred, color="#5B57E6", linewidth=2.2, label="Predicted RUL")
    plt.plot(steps, err_below, color="#D55E5E", linewidth=1.6, label="Prediction Error")
    plt.fill_between(steps, err_below, 0.0, color="#D55E5E", alpha=0.18)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.85)

    plt.xlim(0, max(len(steps) - 1, 1))
    plt.ylim(y_bottom, y_top)
    plt.xlabel("Time Step")
    plt.ylabel("Normalized RUL / Error")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_adjacency_heatmap(adj: torch.Tensor, save_path: str, title: str = "Adaptive Adjacency") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    adj_np = adj.detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(adj_np, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Neighbor node")
    plt.ylabel("Current node")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
