
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BaseConfig:
    """Common configuration for Bi-KAGN experiments."""

    data_root: str = "./data"
    device: str = field(default_factory=default_device)
    seed: int = 42

    # Ablation switches
    use_graphkan: bool = True
    regressor_type: str = "kan"  # "kan", "mlp", or "multkan"

    # Dynamic degradation graph construction
    ddwd_layers: int = 3
    avg_pool_dim: int = 200
    tau: float = 0.6
    graph_dim: int = 6
    k_neighbors: int = 160

    # Bidirectional graph Kolmogorov-Arnold fusion network
    num_heads: int = 4
    gkan_order1: int = 2
    gkan_order2: int = 3
    gkan_norm: str = "rw"
    tcn_layers: int = 5
    tcn_kernel_size: int = 5

    # Memory-augmented Kolmogorov-Arnold prediction network
    memory_size: int = 128
    memory_dim: int = 64
    memory_topk: int = 5
    memory_momentum: float = 0.9
    fc_dim: int = 200

    # Training
    epochs: int = 300
    batch_size: int = 2
    lr: float = 1e-3

    # Loss weights
    alpha_orth: float = 0.7
    alpha_energy: float = 0.3
    alpha_kan_reg: float = 1e-4

    # Wavelet assumptions
    wavelet_kernel_size: int = 8
    eps: float = 1e-8

    # Output
    output_root: str = "./outputs"
    run_name: Optional[str] = None

    def make_run_dir(self, prefix: str) -> Path:
        stamp = self.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.output_root) / prefix / stamp
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class XJTUConfig(BaseConfig):
    """Default settings for the XJTU-SY bearing dataset."""

    data_root: str = "./XJTU"
    output_root: str = "./xjtu_outputs"

    # Paper-style XJTU-SY settings
    avg_pool_dim: int = 200
    tau: float = 0.6
    k_neighbors: int = 160
    memory_size: int = 128
    memory_dim: int = 64
    batch_size: int = 2


@dataclass
class PHM2012Config(BaseConfig):
    """Default settings for the IEEE PHM 2012 / PRONOSTIA bearing dataset."""

    data_root: str = "./ieee-phm-2012-data-challenge-dataset-master"
    train_folder: str = "Learning_set"
    test_folder: str = "Full_Test_Set"
    expected_seq_len: int = 2560
    output_root: str = "./phm2012_outputs"
    explanation_root: str = "./explanations_phm2012"

    # Paper-style PHM2012 settings
    avg_pool_dim: int = 80
    tau: float = 0.5
    k_neighbors: int = 110
    memory_size: int = 64
    memory_dim: int = 64
    batch_size: int = 8
