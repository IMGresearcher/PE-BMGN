
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .configs import BaseConfig
from .model import BiKAGN


def compute_total_loss(
    model: BiKAGN,
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: BaseConfig,
) -> Tuple[torch.Tensor, dict]:
    """Compute prediction, physical-consistency and KAN regularization losses."""
    mse = F.mse_loss(pred, target)
    l_orth, l_energy = model.physical_loss()
    l_kan = model.kan_regularization_loss()

    total = (
        mse
        + cfg.alpha_orth * l_orth
        + cfg.alpha_energy * l_energy
        + cfg.alpha_kan_reg * l_kan
    )

    stat = {
        "mse": float(mse.detach().cpu()),
        "orth": float(l_orth.detach().cpu()),
        "energy": float(l_energy.detach().cpu()),
        "kan": float(l_kan.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, stat
