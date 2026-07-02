
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan_layers import KANLinear, KanChebConv


def module_kan_regularization(module: nn.Module, device: torch.device) -> torch.Tensor:
    """Sum regularization_loss() over all KANLinear modules."""
    loss = None
    for m in module.modules():
        if isinstance(m, KANLinear):
            term = m.regularization_loss()
            loss = term if loss is None else loss + term
    if loss is None:
        return torch.tensor(0.0, device=device)
    return loss


def masked_dense_adj_to_edge_index(
    adjacency: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a masked dense adjacency matrix into PyG edge_index/edge_weight."""
    adj_masked = adjacency * mask.float()

    num_nodes = adj_masked.size(0)
    eye = torch.eye(num_nodes, device=adjacency.device, dtype=adjacency.dtype) * 1e-6
    adj_masked = adj_masked + eye
    adj_masked = adj_masked / (adj_masked.sum(dim=-1, keepdim=True) + eps)

    nonzero = (adj_masked > 0).nonzero(as_tuple=False)
    edge_index = nonzero.t().contiguous()
    edge_weight = adj_masked[nonzero[:, 0], nonzero[:, 1]].contiguous()
    return edge_index, edge_weight


class MultiScaleProjector(nn.Module):
    """Aggregate wavelet coefficients into compact node features."""

    def __init__(self, num_layers: int = 3, out_dim: int = 200):
        super().__init__()
        self.num_layers = num_layers
        self.out_dim = out_dim

    def forward(self, ddwd_out) -> torch.Tensor:
        parts = []
        for c_a, c_d in zip(ddwd_out["cA_list"], ddwd_out["cD_list"]):
            parts.append(c_a.flatten(1))
            parts.append(c_d.flatten(1))

        x_cat = torch.cat(parts, dim=-1)
        x = F.adaptive_avg_pool1d(x_cat.unsqueeze(1), self.out_dim).squeeze(1)
        return x


class AdaptiveAdjacency(nn.Module):
    """Construct sparse adaptive degradation graph topology."""

    def __init__(self, in_dim: int, proj_dim: int, tau: float = 0.6, k_neighbors: int = 160):
        super().__init__()
        self.wq = nn.Linear(in_dim, proj_dim, bias=False)
        self.wk = nn.Linear(in_dim, proj_dim, bias=False)
        self.tau = tau
        self.k_neighbors = k_neighbors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.wq(x), p=2, dim=-1)
        k = F.normalize(self.wk(x), p=2, dim=-1)

        sim = q @ k.T
        adjacency = F.softmax(sim / self.tau, dim=-1)

        num_nodes = adjacency.size(0)
        topk = min(self.k_neighbors, num_nodes)
        values, indices = torch.topk(adjacency, k=topk, dim=-1)
        sparse_adj = torch.zeros_like(adjacency)
        sparse_adj.scatter_(1, indices, values)
        sparse_adj = sparse_adj / (sparse_adj.sum(dim=-1, keepdim=True) + 1e-8)
        return sparse_adj


class GatedFusion(nn.Module):
    """Gated cross fusion of forward and backward graph features."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 3, dim)

    def forward(self, h_fwd: torch.Tensor, h_bwd: torch.Tensor) -> torch.Tensor:
        interaction = h_fwd * h_bwd
        gamma = torch.sigmoid(self.gate(torch.cat([h_fwd, h_bwd, interaction], dim=-1)))
        return (1.0 - gamma) * h_fwd + gamma * h_bwd


class DilatedCausalConvBlock(nn.Module):
    """Temporal smoothing block after graph propagation."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, C]
        x_t = x.transpose(0, 1).unsqueeze(0)
        pad = (self.kernel_size - 1) * self.dilation
        x_t = F.pad(x_t, (pad, 0))
        out = self.conv(x_t).squeeze(0).transpose(0, 1)
        out = F.relu(out)
        return self.norm(out)


class SimpleMaskedMultiHeadGAT(nn.Module):
    """Masked multi-head GAT baseline for ablation."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.wq = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_heads)])
        self.wk = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_heads)])
        self.wv = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_heads)])
        self.attn = nn.ModuleList([nn.Linear(2 * out_dim, 1, bias=False) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        valid_mask = (adjacency > 0) & causal_mask
        head_outputs = []

        for h in range(self.num_heads):
            q = self.wq[h](x)
            k = self.wk[h](x)
            v = self.wv[h](x)

            q_expand = q.unsqueeze(1).repeat(1, num_nodes, 1)
            k_expand = k.unsqueeze(0).repeat(num_nodes, 1, 1)
            score = self.attn[h](torch.cat([q_expand, k_expand], dim=-1)).squeeze(-1)

            score = score.masked_fill(~valid_mask, -1e9)
            alpha = F.softmax(score, dim=-1)
            alpha = alpha * adjacency
            alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-8)
            head_outputs.append(alpha @ v)

        return torch.cat(head_outputs, dim=-1)

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


class BidirectionalGATFusionNetwork(nn.Module):
    """Bidirectional GAT baseline with gated fusion and TCN refinement."""

    def __init__(self, in_dim: int, graph_dim: int, num_heads: int, tcn_layers: int, tcn_kernel_size: int):
        super().__init__()
        self.gat_forward = SimpleMaskedMultiHeadGAT(in_dim, graph_dim, num_heads=num_heads)
        self.gat_backward = SimpleMaskedMultiHeadGAT(in_dim, graph_dim, num_heads=num_heads)
        self.fused_dim = graph_dim * num_heads
        self.fusion = GatedFusion(self.fused_dim)
        self.tcn = nn.ModuleList([
            DilatedCausalConvBlock(
                channels=self.fused_dim,
                kernel_size=tcn_kernel_size,
                dilation=2 ** layer_idx,
            )
            for layer_idx in range(tcn_layers)
        ])

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        idx = torch.arange(num_nodes, device=x.device)
        forward_mask = idx.view(-1, 1) >= idx.view(1, -1)
        backward_mask = idx.view(-1, 1) <= idx.view(1, -1)

        h_fwd = self.gat_forward(x, adjacency, causal_mask=forward_mask)
        h_bwd = self.gat_backward(x, adjacency, causal_mask=backward_mask)

        h = self.fusion(h_fwd, h_bwd)
        for block in self.tcn:
            h = block(h)
        return h

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


class GraphKANBlock(nn.Module):
    """Two-layer Chebyshev graph Kolmogorov-Arnold block."""

    def __init__(self, in_dim: int, hidden_dim: int, order1: int = 2, order2: int = 3, normalization: str = "rw"):
        super().__init__()
        self.conv1 = KanChebConv(in_dim, hidden_dim, K=order1, normalization=normalization)
        self.conv2 = KanChebConv(hidden_dim, hidden_dim, K=order2, normalization=normalization)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(self.norm1(h))
        h = self.conv2(h, edge_index, edge_weight)
        return self.norm2(h)

    def regularization_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        return module_kan_regularization(self, device)


class BidirectionalGraphKANFusionNetwork(nn.Module):
    """Bidirectional graph Kolmogorov-Arnold fusion network."""

    def __init__(
        self,
        in_dim: int,
        graph_dim: int,
        num_heads: int,
        order1: int,
        order2: int,
        normalization: str,
        tcn_layers: int,
        tcn_kernel_size: int,
    ):
        super().__init__()
        self.fused_dim = graph_dim * num_heads

        self.gkan_forward = GraphKANBlock(
            in_dim=in_dim,
            hidden_dim=self.fused_dim,
            order1=order1,
            order2=order2,
            normalization=normalization,
        )
        self.gkan_backward = GraphKANBlock(
            in_dim=in_dim,
            hidden_dim=self.fused_dim,
            order1=order1,
            order2=order2,
            normalization=normalization,
        )

        self.fusion = GatedFusion(self.fused_dim)
        self.tcn = nn.ModuleList([
            DilatedCausalConvBlock(
                channels=self.fused_dim,
                kernel_size=tcn_kernel_size,
                dilation=2 ** layer_idx,
            )
            for layer_idx in range(tcn_layers)
        ])

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        idx = torch.arange(num_nodes, device=x.device)

        forward_mask = idx.view(-1, 1) >= idx.view(1, -1)
        backward_mask = idx.view(-1, 1) <= idx.view(1, -1)

        edge_index_f, edge_weight_f = masked_dense_adj_to_edge_index(adjacency, forward_mask)
        edge_index_b, edge_weight_b = masked_dense_adj_to_edge_index(adjacency, backward_mask)

        h_fwd = self.gkan_forward(x, edge_index_f, edge_weight_f)
        h_bwd = self.gkan_backward(x, edge_index_b, edge_weight_b)

        h = self.fusion(h_fwd, h_bwd)
        for block in self.tcn:
            h = block(h)
        return h

    def regularization_loss(self) -> torch.Tensor:
        return self.gkan_forward.regularization_loss() + self.gkan_backward.regularization_loss()


# Backward-compatible aliases.
BiAGCN = BidirectionalGATFusionNetwork
BiAGKAN = BidirectionalGraphKANFusionNetwork
