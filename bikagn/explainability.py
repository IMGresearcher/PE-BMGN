
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .graph import BidirectionalGraphKANFusionNetwork
from .kan_layers import KANLinear
from .regressors import KANRegressionHead


def register_kan_hooks(model: nn.Module):
    """Register forward hooks to cache KAN layer inputs and outputs."""
    handles = []

    def _hook(module, inputs, output):
        x = inputs[0]
        if isinstance(x, tuple):
            x = x[0]
        if torch.is_tensor(x):
            module.last_input = x.detach().reshape(-1, x.shape[-1]).cpu()
        if torch.is_tensor(output):
            module.last_output = output.detach().reshape(-1, output.shape[-1]).cpu()

    for module in model.modules():
        if isinstance(module, KANLinear):
            handles.append(module.register_forward_hook(_hook))
    return handles


def remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def kan_importance_matrix(kan_layer: KANLinear) -> torch.Tensor:
    with torch.no_grad():
        return kan_layer.scaled_spline_weight.detach().abs().mean(dim=-1).cpu()


def topk_kan_pairs(kan_layer: KANLinear, topk: int = 5):
    imp = kan_importance_matrix(kan_layer)
    flat = imp.flatten()
    values, indices = torch.topk(flat, k=min(topk, flat.numel()))
    pairs = []
    in_dim = imp.size(1)
    for value, index in zip(values.tolist(), indices.tolist()):
        out_idx = index // in_dim
        in_idx = index % in_dim
        pairs.append((int(out_idx), int(in_idx), float(value)))
    return pairs


def sample_kan_curve(
    kan_layer: KANLinear,
    in_idx: int,
    out_idx: int,
    num_points: int = 200,
    ref_x: torch.Tensor = None,
):
    device = next(kan_layer.parameters()).device
    if ref_x is None:
        ref_x = getattr(kan_layer, "last_input", torch.zeros(32, kan_layer.in_features))

    if not torch.is_tensor(ref_x):
        ref_x = torch.tensor(ref_x, dtype=torch.float32)
    ref_x = ref_x.to(device=device, dtype=torch.float32)

    x_min = ref_x[:, in_idx].min().item()
    x_max = ref_x[:, in_idx].max().item()
    if abs(x_max - x_min) < 1e-6:
        x_min -= 1.0
        x_max += 1.0

    xs = torch.linspace(x_min, x_max, num_points, device=device)
    base = ref_x.mean(dim=0, keepdim=True).repeat(num_points, 1)
    base[:, in_idx] = xs

    with torch.no_grad():
        ys = kan_layer(base)[:, out_idx]
    return xs.detach().cpu().numpy(), ys.detach().cpu().numpy()


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def explain_graph_propagation(model, save_dir: str, topk: int = 5, num_points: int = 200) -> None:
    """Visualize top KAN functions in forward/backward GraphKAN propagation."""
    if not isinstance(getattr(model, "graph_backbone", None), BidirectionalGraphKANFusionNetwork):
        return

    os.makedirs(save_dir, exist_ok=True)

    blocks = {
        "forward_conv1": model.graph_backbone.gkan_forward.conv1,
        "forward_conv2": model.graph_backbone.gkan_forward.conv2,
        "backward_conv1": model.graph_backbone.gkan_backward.conv1,
        "backward_conv2": model.graph_backbone.gkan_backward.conv2,
    }

    summary = {}
    for block_name, conv in blocks.items():
        block_summary = {}
        for order_idx, lin in enumerate(conv.lins):
            pairs = topk_kan_pairs(lin, topk=topk)
            block_summary[f"order_{order_idx}"] = pairs
            ref_x = getattr(lin, "last_input", None)

            for rank, (out_idx, in_idx, score) in enumerate(pairs):
                xs, ys = sample_kan_curve(
                    lin,
                    in_idx=in_idx,
                    out_idx=out_idx,
                    num_points=num_points,
                    ref_x=ref_x,
                )
                plt.figure(figsize=(5, 3))
                plt.plot(xs, ys, linewidth=2)
                plt.xlabel(f"input dim {in_idx}")
                plt.ylabel(f"output dim {out_idx}")
                plt.title(f"{block_name} | order {order_idx} | score={score:.4f}")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{block_name}_order{order_idx}_rank{rank}.png"), dpi=220)
                plt.close()

        summary[block_name] = block_summary

    save_json(summary, os.path.join(save_dir, "graph_propagation_summary.json"))


def explain_output_mapping(model, save_dir: str, topk: int = 8, num_points: int = 200) -> None:
    """Visualize top KAN functions in the final KAN regression head."""
    if not isinstance(model.regressor, KANRegressionHead):
        return

    os.makedirs(save_dir, exist_ok=True)

    summary = {}
    layers = [("kan1", model.regressor.kan1), ("kan2", model.regressor.kan2)]
    for layer_name, layer in layers:
        pairs = topk_kan_pairs(layer, topk=topk)
        summary[layer_name] = pairs
        ref_x = getattr(layer, "last_input", None)

        for rank, (out_idx, in_idx, score) in enumerate(pairs):
            xs, ys = sample_kan_curve(
                layer,
                in_idx=in_idx,
                out_idx=out_idx,
                num_points=num_points,
                ref_x=ref_x,
            )
            plt.figure(figsize=(5, 3))
            plt.plot(xs, ys, linewidth=2)
            plt.xlabel(f"input dim {in_idx}")
            plt.ylabel(f"output dim {out_idx}")
            plt.title(f"{layer_name} | score={score:.4f}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{layer_name}_rank{rank}.png"), dpi=220)
            plt.close()

    save_json(summary, os.path.join(save_dir, "output_mapping_summary.json"))
