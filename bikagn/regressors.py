
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import module_kan_regularization
from .kan_layers import KANLinear


class KANRegressionHead(nn.Module):
    """Final KAN-based regressor for function-level interpretability."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.kan1 = KANLinear(in_dim, hidden_dim)
        self.kan2 = KANLinear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.kan1(x))
        return self.kan2(x).squeeze(-1)

    def regularization_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        return module_kan_regularization(self, device)


class MLPRegressionHead(nn.Module):
    """MLP regression head for ablation."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


def _extend_grid(grid: torch.Tensor, k_extend: int) -> torch.Tensor:
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
    out = grid
    for _ in range(k_extend):
        out = torch.cat([out[:, [0]] - h, out], dim=1)
        out = torch.cat([out, out[:, [-1]] + h], dim=1)
    return out

def _B_batch(x: torch.Tensor, grid: torch.Tensor, k: int = 0) -> torch.Tensor:
    x = x.unsqueeze(2)
    grid = grid.unsqueeze(0)

    if k == 0:
        value = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)
    else:
        B_km1 = _B_batch(x[:, :, 0], grid[0], k=k - 1)
        left_num = x - grid[:, :, :-(k + 1)]
        left_den = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
        right_num = grid[:, :, k + 1:] - x
        right_den = grid[:, :, k + 1:] - grid[:, :, 1:(-k)]

        value = (
            (left_num / left_den) * B_km1[:, :, :-1]
            + (right_num / right_den) * B_km1[:, :, 1:]
        )

    return torch.nan_to_num(value)

def _coef2curve(x_eval: torch.Tensor, grid: torch.Tensor, coef: torch.Tensor, k: int) -> torch.Tensor:
    b_splines = _B_batch(x_eval, grid, k=k)
    y_eval = torch.einsum("big,iog->bio", b_splines, coef.to(b_splines.device))
    return y_eval

def _curve2coef(x_eval: torch.Tensor, y_eval: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1

    mat = _B_batch(x_eval, grid, k)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)
    y_eval = y_eval.permute(1, 2, 0).unsqueeze(3)

    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution[:, :, :, 0]
    except RuntimeError:
        Xt = mat.transpose(-1, -2)
        eye = torch.eye(n_coef, device=mat.device, dtype=mat.dtype)[None, None, :, :]
        coef = torch.linalg.solve(
            Xt @ mat + 1e-6 * eye,
            Xt @ y_eval
        )[:, :, :, 0]

    return coef

class _MiniKANLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num: int = 5,
        k: int = 3,
        noise_scale: float = 0.1,
        scale_base_mu: float = 0.0,
        scale_base_sigma: float = 1.0,
        scale_sp: float = 1.0,
        base_fun: nn.Module = None,
        grid_eps: float = 0.02,
        grid_range = (-1.0, 1.0),
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        sparse_init: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = num
        self.k = k
        self.grid_eps = grid_eps
        self.device = device

        if base_fun is None:
            base_fun = nn.SiLU()
        self.base_fun = base_fun

        base_grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(in_dim, num + 1)
        base_grid = _extend_grid(base_grid, k_extend=k)
        self.grid = nn.Parameter(base_grid, requires_grad=False)

        noises = (torch.rand(num + 1, in_dim, out_dim) - 0.5) * noise_scale / max(num, 1)
        coef_init = _curve2coef(
            self.grid[:, k:-k].permute(1, 0),
            noises,
            self.grid,
            k
        )
        self.coef = nn.Parameter(coef_init)

        if sparse_init:
            mask = (torch.rand(in_dim, out_dim) > 0.5).float()
            if mask.sum() == 0:
                mask[0, 0] = 1.0
        else:
            mask = torch.ones(in_dim, out_dim)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.scale_base = nn.Parameter(
            scale_base_mu * (1.0 / math.sqrt(max(in_dim, 1))) +
            scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) * (1.0 / math.sqrt(max(in_dim, 1))),
            requires_grad=sb_trainable
        )
        self.scale_sp = nn.Parameter(
            torch.ones(in_dim, out_dim) * scale_sp * (1.0 / math.sqrt(max(in_dim, 1))) * self.mask,
            requires_grad=sp_trainable
        )

        self.to(device)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        preacts = x[:, None, :].expand(B, self.out_dim, self.in_dim)

        base = self.base_fun(x)
        y_sp = _coef2curve(x, self.grid, self.coef, self.k)
        postspline = y_sp.permute(0, 2, 1).contiguous()

        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * y_sp
        y = self.mask[None, :, :] * y

        postacts = y.permute(0, 2, 1).contiguous()
        y = y.sum(dim=1)
        return y, preacts, postacts, postspline

class _MiniMultKANLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_sum: int,
        out_mult: int,
        mult_arity: int = 2,
        grid: int = 5,
        k: int = 3,
        noise_scale: float = 0.1,
        base_fun: nn.Module = None,
        sparse_init: bool = False,
        affine_trainable: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        assert mult_arity >= 2
        assert out_sum >= 0 and out_mult >= 0
        assert out_sum + out_mult > 0

        self.in_dim = in_dim
        self.out_sum = out_sum
        self.out_mult = out_mult
        self.mult_arity = mult_arity
        self.out_dim_pre = out_sum + mult_arity * out_mult
        self.out_dim_post = out_sum + out_mult

        self.kan = _MiniKANLayer(
            in_dim=in_dim,
            out_dim=self.out_dim_pre,
            num=grid,
            k=k,
            noise_scale=noise_scale,
            base_fun=base_fun,
            sparse_init=sparse_init,
            device=device,
        )

        self.subnode_bias = nn.Parameter(torch.zeros(self.out_dim_pre), requires_grad=affine_trainable)
        self.subnode_scale = nn.Parameter(torch.ones(self.out_dim_pre), requires_grad=affine_trainable)
        self.node_bias = nn.Parameter(torch.zeros(self.out_dim_post), requires_grad=affine_trainable)
        self.node_scale = nn.Parameter(torch.ones(self.out_dim_post), requires_grad=affine_trainable)

        self.last_edge_scale = None
        self.last_postspline_scale = None

    def forward(self, x: torch.Tensor):
        z, _preacts, postacts, postspline = self.kan(x)

        self.last_edge_scale = postacts.abs().mean(dim=0)
        self.last_postspline_scale = postspline.abs().mean(dim=0)

        z = z * self.subnode_scale[None, :] + self.subnode_bias[None, :]

        if self.out_mult > 0:
            sum_part = z[:, :self.out_sum] if self.out_sum > 0 else z[:, :0]
            mul_raw = z[:, self.out_sum:]
            mul_raw = mul_raw.view(z.shape[0], self.out_mult, self.mult_arity)
            mul_part = torch.prod(mul_raw, dim=-1)
            out = torch.cat([sum_part, mul_part], dim=-1)
        else:
            out = z[:, :self.out_sum]

        out = out * self.node_scale[None, :] + self.node_bias[None, :]
        return out

class MultKANRegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()

        if hidden_dim <= 3:
            n_mult = 0
            n_sum = hidden_dim
        else:
            n_mult = max(1, hidden_dim // 4)
            n_sum = hidden_dim - n_mult

        self.grid = 5
        self.k = 3
        self.mult_arity = 2
        self.noise_scale = 0.05
        self.affine_trainable = False
        self.sparse_init = False

        self.reg_lamb_l1 = 1e-3
        self.reg_lamb_entropy = 1e-3
        self.reg_lamb_coef = 1e-3
        self.reg_lamb_coefdiff = 1e-3

        self.layer1 = _MiniMultKANLayer(
            in_dim=in_dim,
            out_sum=n_sum,
            out_mult=n_mult,
            mult_arity=self.mult_arity,
            grid=self.grid,
            k=self.k,
            noise_scale=self.noise_scale,
            base_fun=nn.SiLU(),
            sparse_init=self.sparse_init,
            affine_trainable=self.affine_trainable,
        )

        self.layer2 = _MiniMultKANLayer(
            in_dim=hidden_dim,
            out_sum=1,
            out_mult=0,
            mult_arity=self.mult_arity,
            grid=self.grid,
            k=self.k,
            noise_scale=self.noise_scale,
            base_fun=nn.SiLU(),
            sparse_init=self.sparse_init,
            affine_trainable=self.affine_trainable,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return x.squeeze(-1)

    def regularization_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        reg = torch.tensor(0.0, device=device)

        edge_scales = [
            self.layer1.last_postspline_scale,
            self.layer2.last_postspline_scale,
        ]

        for vec in edge_scales:
            if vec is None:
                continue
            l1 = torch.sum(vec)

            p_row = vec / (torch.sum(vec, dim=1, keepdim=True) + 1.0)
            p_col = vec / (torch.sum(vec, dim=0, keepdim=True) + 1.0)

            entropy_row = -torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1))
            entropy_col = -torch.mean(torch.sum(p_col * torch.log2(p_col + 1e-4), dim=0))

            reg = reg + self.reg_lamb_l1 * l1
            reg = reg + self.reg_lamb_entropy * (entropy_row + entropy_col)

        kan_layers = [self.layer1.kan, self.layer2.kan]
        for kl in kan_layers:
            coeff_l1 = torch.sum(torch.mean(torch.abs(kl.coef), dim=1))
            coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(kl.coef, dim=-1)), dim=1))
            reg = reg + self.reg_lamb_coef * coeff_l1
            reg = reg + self.reg_lamb_coefdiff * coeff_diff_l1

        return reg