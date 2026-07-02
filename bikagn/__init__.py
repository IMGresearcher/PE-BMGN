"""Bi-KAGN: Bidirectional Kolmogorov-Arnold-informed Graph Neural Network for bearing RUL prediction."""

from .configs import BaseConfig, XJTUConfig, PHM2012Config
from .model import BiKAGN, DyWaveBiAGKAN

__all__ = [
    "BaseConfig",
    "XJTUConfig",
    "PHM2012Config",
    "BiKAGN",
    "DyWaveBiAGKAN",
]
