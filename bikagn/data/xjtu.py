
from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from ..utils import build_rul_labels, natural_key, zscore_apply, zscore_fit_from_runs


def condition_groups() -> Dict[str, List[str]]:
    """XJTU-SY three operating conditions."""
    return {
        "Condition1": [f"Bearing1_{j}" for j in range(1, 6)],
        "Condition2": [f"Bearing2_{j}" for j in range(1, 6)],
        "Condition3": [f"Bearing3_{j}" for j in range(1, 6)],
    }


def default_bearing_names() -> List[str]:
    names: List[str] = []
    for group_names in condition_groups().values():
        names.extend(group_names)
    return names


def get_condition_name(test_name: str) -> str:
    for cond_name, group_names in condition_groups().items():
        if test_name in group_names:
            return cond_name
    raise ValueError(f"Unknown bearing name: {test_name}")


def get_condition_bearing_names(test_name: str) -> List[str]:
    return condition_groups()[get_condition_name(test_name)]


def load_one_csv(file_path: str) -> np.ndarray:
    """Load one XJTU-SY CSV file and return [2, L]."""
    df = pd.read_csv(file_path)
    if df.shape[1] < 2:
        df = pd.read_csv(file_path, header=None)

    df = df.iloc[:, :2]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    data = df.values.astype(np.float32)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected shape [L, 2], got {data.shape} in {file_path}")
    return data.T


def load_bearing_run(root: str, bearing_name: str) -> torch.Tensor:
    """Load one complete XJTU-SY bearing run as [T, 2, L]."""
    folder = os.path.join(root, bearing_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = sorted(glob.glob(os.path.join(folder, "*.csv")), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No CSV found in {folder}")

    samples = [torch.from_numpy(load_one_csv(fp)) for fp in files]
    return torch.stack(samples, dim=0)


def prepare_xjtu_runs(data_root: str, bearing_names: List[str] = None):
    if bearing_names is None:
        bearing_names = default_bearing_names()

    run_dict = {}
    label_dict = {}
    for name in bearing_names:
        run_x = load_bearing_run(data_root, name)
        run_dict[name] = run_x
        label_dict[name] = build_rul_labels(run_x.size(0))
    return run_dict, label_dict


def split_leave_one_out(run_dict, label_dict, test_name: str):
    train_runs, train_labels, test_runs, test_labels = {}, {}, {}, {}
    for name in run_dict.keys():
        if name == test_name:
            test_runs[name] = run_dict[name]
            test_labels[name] = label_dict[name]
        else:
            train_runs[name] = run_dict[name]
            train_labels[name] = label_dict[name]
    return train_runs, train_labels, test_runs, test_labels


def normalize_train_test(train_runs: Dict[str, torch.Tensor], test_runs: Dict[str, torch.Tensor]):
    mean, std = zscore_fit_from_runs(list(train_runs.values()))
    train_runs = {k: zscore_apply(v, mean, std) for k, v in train_runs.items()}
    test_runs = {k: zscore_apply(v, mean, std) for k, v in test_runs.items()}
    return train_runs, test_runs
