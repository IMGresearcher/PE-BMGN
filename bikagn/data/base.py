
from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset


class BearingRunDataset(Dataset):
    """One item is one complete bearing run."""

    def __init__(self, run_dict: Dict[str, torch.Tensor], label_dict: Dict[str, torch.Tensor]):
        self.names = list(run_dict.keys())
        self.run_dict = run_dict
        self.label_dict = label_dict

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        return {
            "name": name,
            "x": self.run_dict[name],
            "y": self.label_dict[name],
        }


def collate_run_batch(batch):
    """Keep variable-length bearing runs as a Python list."""
    return batch
