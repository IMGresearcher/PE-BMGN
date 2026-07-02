
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DynamicMemoryBank(nn.Module):
    """Dynamic memory bank for historical degradation prototypes.

    The retrieval operation first ranks all memory prototypes by softmax
    similarity and then performs a normalized top-K weighted aggregation.
    """

    def __init__(
        self,
        feature_dim: int,
        memory_size: int,
        memory_dim: int,
        topk: int = 5,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, memory_dim)
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.topk = topk
        self.momentum = momentum

        memory = torch.randn(memory_size, memory_dim) * 0.02
        self.register_buffer("memory", memory)

    def retrieve(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve memory-enhanced representation.

        Args:
            h: Fused degradation representation with shape [T, feature_dim].

        Returns:
            query: Projected memory query [T, memory_dim].
            h_mem: Weighted top-K memory representation [T, memory_dim].
            attn: Full relevance distribution over memory prototypes [T, F].
        """
        query = self.query_proj(h)
        attn = torch.softmax(query @ self.memory.T, dim=-1)

        k = min(self.topk, self.memory_size)
        top_values, top_indices = torch.topk(attn, k=k, dim=-1)
        selected = self.memory[top_indices]

        top_weights = top_values / (top_values.sum(dim=-1, keepdim=True) + 1e-8)
        h_mem = (selected * top_weights.unsqueeze(-1)).sum(dim=1)
        return query, h_mem, attn

    @torch.no_grad()
    def momentum_update(self, query_batch_mean: torch.Tensor) -> None:
        """Momentum update of the memory prototypes.

        The original implementation updates all prototypes with the mini-batch
        average query. This compact update is retained for reproducibility.
        """
        query_batch_mean = query_batch_mean.view(1, -1)
        self.memory.mul_(self.momentum).add_((1.0 - self.momentum) * query_batch_mean)
