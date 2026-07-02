
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def db4_filters() -> Tuple[np.ndarray, np.ndarray]:
    """Classical DB4 decomposition low-pass and high-pass coefficients."""
    lo = np.array([
        -0.010597401785069032,
         0.0328830116668852,
         0.030841381835560764,
        -0.18703481171888114,
        -0.027983769416859854,
         0.6308807679298587,
         0.7148465705529157,
         0.2303778133088964,
    ], dtype=np.float32)

    hi = np.array([
        -0.2303778133088964,
         0.7148465705529157,
        -0.6308807679298587,
        -0.027983769416859854,
         0.18703481171888114,
         0.030841381835560764,
        -0.0328830116668852,
        -0.010597401785069032,
    ], dtype=np.float32)
    return lo, hi


class LearnableWaveletFilterBank(nn.Module):
    """DB4-initialized learnable low/high filter bank.

    Filters are shared across channels and applied depthwise.
    """

    def __init__(self, channels: int, kernel_size: int = 8):
        super().__init__()
        lo, hi = db4_filters()
        if len(lo) != kernel_size:
            raise ValueError(f"DB4 has length {len(lo)}, got kernel_size={kernel_size}.")
        if kernel_size % 2 != 0:
            raise ValueError("This implementation expects an even wavelet kernel size.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.theta_lo = nn.Parameter(torch.tensor(lo).view(1, 1, -1))
        self.theta_hi = nn.Parameter(torch.tensor(hi).view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return approximation/detail coefficients for input [T, C, L]."""
        _, channels, seq_len = x.shape
        pad_each = self.kernel_size // 2 - 1
        x_pad = F.pad(x, (pad_each, pad_each), mode="reflect")

        lo = self.theta_lo.repeat(channels, 1, 1)
        hi = self.theta_hi.repeat(channels, 1, 1)

        c_a = F.conv1d(x_pad, lo, stride=2, groups=channels)
        c_d = F.conv1d(x_pad, hi, stride=2, groups=channels)

        expected_len = seq_len // 2
        c_a = c_a[..., :expected_len]
        c_d = c_d[..., :expected_len]
        return c_a, c_d


class FrequencyGate(nn.Module):
    """Gate spectral amplitude while preserving phase."""

    def __init__(self, channels: int, pooled_len: int = 64):
        super().__init__()
        self.pooled_len = pooled_len
        self.wg = nn.Conv1d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = torch.fft.rfft(x, dim=-1)
        amp = torch.abs(x_f)
        phase = torch.angle(x_f)

        pooled = F.adaptive_avg_pool1d(amp, self.pooled_len)
        gate_small = torch.sigmoid(self.wg(pooled))
        gate = F.interpolate(gate_small, size=amp.shape[-1], mode="linear", align_corners=False)

        amp_hat = gate * amp
        real = amp_hat * torch.cos(phase)
        imag = amp_hat * torch.sin(phase)
        x_f_hat = torch.complex(real, imag)
        return torch.fft.irfft(x_f_hat, n=x.shape[-1], dim=-1)


class MaxPoolResidualEnhance(nn.Module):
    """Max-pooling residual enhancement for dominant temporal responses."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        return x + pooled


class LearnableWaveletDecomposition(nn.Module):
    """Recursive learnable discrete wavelet decomposition.

    The approximation branch is enhanced by max-pooling residuals and used as
    the next-level input. The detail branch is enhanced by frequency-domain
    gating.
    """

    def __init__(self, channels: int = 2, num_layers: int = 3, kernel_size: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.filter_bank = LearnableWaveletFilterBank(channels=channels, kernel_size=kernel_size)
        self.freq_gate = FrequencyGate(channels=channels)
        self.approx_enhance = MaxPoolResidualEnhance(kernel_size=3)

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        c_a_list: List[torch.Tensor] = []
        c_d_list: List[torch.Tensor] = []

        current = x
        for _ in range(self.num_layers):
            c_a, c_d = self.filter_bank(current)
            c_a_hat = self.approx_enhance(c_a)
            c_d_hat = self.freq_gate(c_d)

            c_a_list.append(c_a_hat)
            c_d_list.append(c_d_hat)
            current = c_a_hat

        return {"cA_list": c_a_list, "cD_list": c_d_list}

    def orthogonality_loss(self) -> torch.Tensor:
        lo = self.filter_bank.theta_lo.squeeze(0).squeeze(0)
        hi = self.filter_bank.theta_hi.squeeze(0).squeeze(0).flip(0)

        conv = F.conv1d(
            lo.view(1, 1, -1),
            hi.view(1, 1, -1),
            padding=hi.numel() - 1,
        ).view(-1)

        target = torch.zeros_like(conv)
        target[conv.numel() // 2] = 1.0
        return F.mse_loss(conv, target)

    def energy_loss(self) -> torch.Tensor:
        lo = self.filter_bank.theta_lo.squeeze()
        hi = self.filter_bank.theta_hi.squeeze()
        return (lo.norm(p=2) ** 2 - 1.0) ** 2 + (hi.norm(p=2) ** 2 - 1.0) ** 2


# Backward-compatible aliases.
DDWD = LearnableWaveletDecomposition
LowFreqEnhance = MaxPoolResidualEnhance
