
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .configs import BaseConfig
from .losses import compute_total_loss
from .utils import eas, mae, rmse
from .visualization import save_rul_comparison_plot


def train_one_epoch(model, loader, optimizer, device: str, cfg: BaseConfig):
    """Train one epoch.

    Because one sample is one full bearing run and runs can have different
    lengths, each DataLoader batch is a list of run dictionaries.
    """
    model.train()
    stats = []

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        batch_loss = 0.0
        batch_query_means = []

        for item in batch:
            x = item["x"].to(device, non_blocking=True)
            y = item["y"].to(device, non_blocking=True)

            out = model(x)
            loss, stat = compute_total_loss(model, out["pred"], y, cfg)
            batch_loss = batch_loss + loss
            batch_query_means.append(out["query"].mean(dim=0).detach())
            stats.append(stat)

        batch_loss = batch_loss / max(len(batch), 1)
        batch_loss.backward()
        optimizer.step()

        with torch.no_grad():
            if batch_query_means:
                query_mean = torch.stack(batch_query_means, dim=0).mean(dim=0)
                model.memory_bank.momentum_update(query_mean)

    if not stats:
        return {"mse": 0.0, "orth": 0.0, "energy": 0.0, "kan": 0.0, "total": 0.0}
    return {key: float(np.mean([s[key] for s in stats])) for key in stats[0].keys()}


@torch.no_grad()
def evaluate_per_bearing(
    model,
    loader,
    device: str,
    plot_dir: Optional[str] = None,
    plot_title_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluate each bearing run separately."""
    model.eval()
    rows = []

    for batch in loader:
        for item in batch:
            name = item["name"]
            x = item["x"].to(device, non_blocking=True)
            y = item["y"].to(device, non_blocking=True)

            out = model(x)
            pred = out["pred"].detach().cpu().numpy()
            target = y.detach().cpu().numpy()

            row = {
                "Bearing": name,
                "RMSE": rmse(pred, target),
                "MAE": mae(pred, target),
                "EAS": eas(pred, target),
            }

            if plot_dir is not None:
                plot_path = Path(plot_dir) / f"{name}_rul.png"
                save_rul_comparison_plot(
                    pred,
                    target,
                    name,
                    str(plot_path),
                    title_prefix=plot_title_prefix,
                )
                row["PlotPath"] = str(plot_path)

            rows.append(row)

    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate(model, loader, device: str):
    """Return mean metrics over bearing-level metrics."""
    df = evaluate_per_bearing(model, loader, device)
    return {
        "RMSE": float(df["RMSE"].mean()),
        "MAE": float(df["MAE"].mean()),
        "EAS": float(df["EAS"].mean()),
    }
