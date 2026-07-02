
from __future__ import annotations

import glob
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch

from ..utils import build_rul_labels, natural_key, zscore_apply, zscore_fit_from_runs


PHM2012_CONDITIONS = {
    "Condition_I": {
        "train": ["Bearing1_1", "Bearing1_2"],
        "test": ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7"],
    },
    "Condition_II": {
        "train": ["Bearing2_1", "Bearing2_2"],
        "test": ["Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7"],
    },
    "Condition_III": {
        "train": ["Bearing3_1", "Bearing3_2"],
        "test": ["Bearing3_3"],
    },
}


def load_one_csv(file_path: str) -> np.ndarray:
    """Load one PHM2012/PRONOSTIA acc_*.csv file and return [2, L]."""
    df = None
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception:
        df = None

    if df is None or df.shape[1] < 2:
        try:
            df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        except Exception:
            pass

    if df is not None and df.shape[1] == 1:
        first_col = df.iloc[:, 0].astype(str)
        if first_col.str.contains(";").any():
            df = first_col.str.split(";", expand=True)
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna(axis=0, how="any")
            df = df.dropna(axis=1, how="all")

    if df is None:
        raise ValueError(f"Unable to read file: {file_path}")

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="any")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise ValueError(f"File {file_path} has fewer than 2 numeric columns: {df.shape}")

    if df.shape[1] >= 5:
        data = df.iloc[:, -2:].values.astype(np.float32)
    else:
        data = df.iloc[:, :2].values.astype(np.float32)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected shape [L, 2], got {data.shape} in {file_path}")
    return data.T


def load_bearing_run(subset_root: str, bearing_name: str, expected_seq_len: int = 2560) -> torch.Tensor:
    """Load one complete PHM2012 bearing run as [T, 2, L]."""
    folder = os.path.join(subset_root, bearing_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = sorted(glob.glob(os.path.join(folder, "acc_*.csv")), key=natural_key)
    if not files:
        preview = sorted(os.listdir(folder))[:20]
        raise RuntimeError(f"No acc files found in {folder}. First items: {preview}")

    samples = []
    bad_files = []
    for fp in files:
        try:
            x = load_one_csv(fp)
            if x.shape[1] != expected_seq_len:
                bad_files.append((fp, x.shape))
                continue
            samples.append(torch.from_numpy(x))
        except Exception as exc:
            bad_files.append((fp, str(exc)))

    if not samples:
        raise RuntimeError(f"No valid acc samples found in {folder}. First failures: {bad_files[:10]}")

    if bad_files:
        print(f"[Warning] {bearing_name}: skipped {len(bad_files)} abnormal acc files")
        for item in bad_files[:10]:
            print("  ", item)

    return torch.stack(samples, dim=0)


def prepare_phm2012_condition_runs(
    data_root: str,
    train_folder: str,
    test_folder: str,
    condition_name: str,
    expected_seq_len: int = 2560,
):
    train_root = os.path.join(data_root, train_folder)
    test_root = os.path.join(data_root, test_folder)

    if condition_name not in PHM2012_CONDITIONS:
        raise KeyError(f"Unknown condition: {condition_name}")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Missing folder: {train_root}")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Missing folder: {test_root}")

    cond_spec = PHM2012_CONDITIONS[condition_name]
    train_runs, train_labels, test_runs, test_labels = {}, {}, {}, {}

    for name in cond_spec["train"]:
        run_x = load_bearing_run(train_root, name, expected_seq_len=expected_seq_len)
        train_runs[name] = run_x
        train_labels[name] = build_rul_labels(run_x.size(0))

    for name in cond_spec["test"]:
        run_x = load_bearing_run(test_root, name, expected_seq_len=expected_seq_len)
        test_runs[name] = run_x
        test_labels[name] = build_rul_labels(run_x.size(0))

    return train_runs, train_labels, test_runs, test_labels


def normalize_train_test(train_runs: Dict[str, torch.Tensor], test_runs: Dict[str, torch.Tensor]):
    mean, std = zscore_fit_from_runs(list(train_runs.values()))
    train_runs = {k: zscore_apply(v, mean, std) for k, v in train_runs.items()}
    test_runs = {k: zscore_apply(v, mean, std) for k, v in test_runs.items()}
    return train_runs, test_runs
