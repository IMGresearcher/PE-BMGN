#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bikagn.configs import XJTUConfig
from bikagn.data.base import BearingRunDataset, collate_run_batch
from bikagn.data.xjtu import (
    condition_groups,
    get_condition_bearing_names,
    get_condition_name,
    normalize_train_test,
    prepare_xjtu_runs,
    split_leave_one_out,
)
from bikagn.model import BiKAGN
from bikagn.training import evaluate, evaluate_per_bearing, train_one_epoch
from bikagn.utils import set_seed


def ablation_name(cfg: XJTUConfig) -> str:
    graph_name = "GraphKAN" if cfg.use_graphkan else "GAT"
    return f"{graph_name}+{cfg.regressor_type.upper()}"


def run_one_fold(cfg: XJTUConfig, test_name: str, result_root: Path):
    cond_name = get_condition_name(test_name)
    cond_bearings = get_condition_bearing_names(test_name)

    print(f"\n========== {cond_name} | LOO Test: {test_name} ==========")
    print(f"Ablation: {ablation_name(cfg)}")
    print(f"Train/Test are restricted to: {cond_bearings}")

    run_dict, label_dict = prepare_xjtu_runs(cfg.data_root, cond_bearings)
    train_runs, train_labels, test_runs, test_labels = split_leave_one_out(run_dict, label_dict, test_name)
    train_runs, test_runs = normalize_train_test(train_runs, test_runs)

    train_ds = BearingRunDataset(train_runs, train_labels)
    test_ds = BearingRunDataset(test_runs, test_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_run_batch,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_run_batch,
        pin_memory=torch.cuda.is_available(),
    )

    model = BiKAGN(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_rmse = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        train_stat = train_one_epoch(model, train_loader, optimizer, cfg.device, cfg)
        eval_stat = evaluate(model, test_loader, cfg.device)

        if eval_stat["RMSE"] < best_rmse:
            best_rmse = eval_stat["RMSE"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"TrainLoss={train_stat['total']:.6f} "
                f"MSE={train_stat['mse']:.6f} "
                f"Orth={train_stat['orth']:.6f} "
                f"Energy={train_stat['energy']:.6f} "
                f"KAN={train_stat['kan']:.6f} | "
                f"RMSE={eval_stat['RMSE']:.6f} "
                f"MAE={eval_stat['MAE']:.6f} "
                f"EAS={eval_stat['EAS']:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    plot_dir = result_root / ablation_name(cfg) / cond_name / "rul_plots"
    final_df = evaluate_per_bearing(
        model,
        test_loader,
        cfg.device,
        plot_dir=str(plot_dir),
        plot_title_prefix=cond_name,
    )
    final_stat = {
        "RMSE": float(final_df["RMSE"].mean()),
        "MAE": float(final_df["MAE"].mean()),
        "EAS": float(final_df["EAS"].mean()),
    }
    print(f"[Best] {cond_name} | {test_name} | {ablation_name(cfg)}: {final_stat}")
    return final_stat


def run_all_loo(cfg: XJTUConfig):
    set_seed(cfg.seed)
    result_root = cfg.make_run_dir("xjtu")
    results = []

    print(f"\nRunning setting: {ablation_name(cfg)}")
    print(f"Output directory: {result_root}")

    for cond_name, bearing_list in condition_groups().items():
        print(f"\n==================== {cond_name} ====================")
        for name in bearing_list:
            stat = run_one_fold(cfg, name, result_root)
            row = {
                "Ablation": ablation_name(cfg),
                "Condition": cond_name,
                "Bearing": name,
                **stat,
            }
            results.append(row)

    df = pd.DataFrame(results)
    save_root = result_root / ablation_name(cfg)
    save_root.mkdir(parents=True, exist_ok=True)

    all_csv = save_root / "xjtu_all_results.csv"
    cond_csv = save_root / "xjtu_condition_mean.csv"

    df.to_csv(all_csv, index=False)
    df.groupby("Condition")[["RMSE", "MAE", "EAS"]].mean().to_csv(cond_csv)

    print("\n========== Overall ==========")
    print(df)
    print("\nAverage by Condition:")
    print(df.groupby("Condition")[["RMSE", "MAE", "EAS"]].mean())
    print("\nAverage over all 15 bearings:")
    print(df[["RMSE", "MAE", "EAS"]].mean())

    print(f"\nSaved all results to: {all_csv}")
    print(f"Saved condition averages to: {cond_csv}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Train Bi-KAGN on XJTU-SY with leave-one-out evaluation.")
    parser.add_argument("--data-root", default="./XJTU")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--regressor", choices=["kan", "mlp", "multkan"], default="kan")
    parser.add_argument("--use-gat", action="store_true", help="Use GAT baseline instead of GraphKAN.")
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = XJTUConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        regressor_type=args.regressor,
        use_graphkan=not args.use_gat,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    if args.device is not None:
        cfg.device = args.device
    run_all_loo(cfg)


if __name__ == "__main__":
    main()
