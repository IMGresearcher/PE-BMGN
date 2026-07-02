
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .configs import BaseConfig
from .graph import (
    AdaptiveAdjacency,
    BidirectionalGATFusionNetwork,
    BidirectionalGraphKANFusionNetwork,
    MultiScaleProjector,
)
from .memory import DynamicMemoryBank
from .regressors import KANRegressionHead, MLPRegressionHead, MultKANRegressionHead
from .wavelet import LearnableWaveletDecomposition


class BiKAGN(nn.Module):
    """Bidirectional Kolmogorov-Arnold-informed graph neural network."""

    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg

        self.ddwd = LearnableWaveletDecomposition(
            channels=2,
            num_layers=cfg.ddwd_layers,
            kernel_size=cfg.wavelet_kernel_size,
        )
        self.projector = MultiScaleProjector(
            num_layers=cfg.ddwd_layers,
            out_dim=cfg.avg_pool_dim,
        )
        self.adj = AdaptiveAdjacency(
            in_dim=cfg.avg_pool_dim,
            proj_dim=cfg.graph_dim,
            tau=cfg.tau,
            k_neighbors=cfg.k_neighbors,
        )

        if cfg.use_graphkan:
            self.graph_backbone = BidirectionalGraphKANFusionNetwork(
                in_dim=cfg.avg_pool_dim,
                graph_dim=cfg.graph_dim,
                num_heads=cfg.num_heads,
                order1=cfg.gkan_order1,
                order2=cfg.gkan_order2,
                normalization=cfg.gkan_norm,
                tcn_layers=cfg.tcn_layers,
                tcn_kernel_size=cfg.tcn_kernel_size,
            )
        else:
            self.graph_backbone = BidirectionalGATFusionNetwork(
                in_dim=cfg.avg_pool_dim,
                graph_dim=cfg.graph_dim,
                num_heads=cfg.num_heads,
                tcn_layers=cfg.tcn_layers,
                tcn_kernel_size=cfg.tcn_kernel_size,
            )

        fused_dim = cfg.graph_dim * cfg.num_heads
        self.memory_bank = DynamicMemoryBank(
            feature_dim=fused_dim,
            memory_size=cfg.memory_size,
            memory_dim=cfg.memory_dim,
            topk=cfg.memory_topk,
            momentum=cfg.memory_momentum,
        )

        reg_type = cfg.regressor_type.lower()
        if reg_type == "kan":
            self.regressor = KANRegressionHead(in_dim=cfg.memory_dim * 2, hidden_dim=cfg.fc_dim)
        elif reg_type == "mlp":
            self.regressor = MLPRegressionHead(in_dim=cfg.memory_dim * 2, hidden_dim=cfg.fc_dim)
        elif reg_type == "multkan":
            self.regressor = MultKANRegressionHead(in_dim=cfg.memory_dim * 2, hidden_dim=cfg.fc_dim)
        else:
            raise ValueError(f"Unsupported regressor_type: {cfg.regressor_type}")

        # Backward-compatible name used in earlier scripts.
        self.fc = self.regressor

    def forward(self, run_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ddwd_out = self.ddwd(run_x)
        node_x = self.projector(ddwd_out)
        adjacency = self.adj(node_x)
        h = self.graph_backbone(node_x, adjacency)
        query, h_mem, memory_attn = self.memory_bank.retrieve(h)
        pred = self.regressor(torch.cat([query, h_mem], dim=-1))

        return {
            "pred": pred,
            "adj": adjacency,
            "node_x": node_x,
            "h": h,
            "query": query,
            "q": query,  # legacy alias
            "h_mem": h_mem,
            "memory_attn": memory_attn,
        }

    def physical_loss(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ddwd.orthogonality_loss(), self.ddwd.energy_loss()

    def kan_regularization_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)

        if self.cfg.use_graphkan:
            loss = loss + self.graph_backbone.regularization_loss()

        if self.cfg.regressor_type.lower() in ("kan", "multkan"):
            loss = loss + self.regressor.regularization_loss()

        return loss


# Backward-compatible alias used by the original scripts.
DyWaveBiAGKAN = BiKAGN
