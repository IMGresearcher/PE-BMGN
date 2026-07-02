#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bikagn.configs import PHM2012Config
from bikagn.data.base import BearingRunDataset, collate_run_batch
from bikagn.data.phm2012 import PHM2012_CONDITIONS, normalize_train_test, prepare_phm2012_condition_runs
from bikagn.explainability import (
    explain_graph_propagation,
    explain_output_mapping,
    register_kan_hooks,
    remove_hooks,
)
from bikagn.model import BiKAGN
from bikagn.training import evaluate, evaluate_per_bearing, train_one_epoch
from bikagn.utils import set_seed
from bikagn.visualization import save_adjacency_heatmap


def ablation_name(cfg: PHM2012Config) -> str:
    graph_name = "GraphKAN" if cfg.use_graphkan else "GAT"
    return f"{graph_name}+{cfg.regressor_type.upper()}"


@torch.no_grad()
def refresh_explanation_cache(model, test_loader, device):
    model.eval()
    for batch in test_loader:
        for item in batch:
            x = item["x"].to(device)
            return model(x)
    return None


def run_condition(cfg: PHM2012Config, condition_name: str, result_root: Path, explain: bool = False):
    print(f"\n========== PHM2012 {condition_name} ==========")
    print(f"Ablation: {ablation_name(cfg)}")

    train_runs, train_labels, test_runs, test_labels = prepare_phm2012_condition_runs(
        cfg.data_root,
        cfg.train_folder,
        cfg.test_folder,
        condition_name,
        expected_seq_len=cfg.expected_seq_len,
    )

    print("Loaded training bearings:")
    for name, tensor in train_runs.items():
        print(f"  {name}: {tuple(tensor.shape)}")
    print("Loaded testing bearings:")
    for name, tensor in test_runs.items():
        print(f"  {name}: {tuple(tensor.shape)}")

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
    hook_handles = register_kan_hooks(model) if explain else []
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
                f"[{condition_name}] Epoch {epoch:03d} | "
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

    if explain:
        first_out = refresh_explanation_cache(model, test_loader, cfg.device)
        explain_dir = Path(cfg.explanation_root) / (cfg.run_name or result_root.name) / ablation_name(cfg) / condition_name
        explain_dir.mkdir(parents=True, exist_ok=True)

        if cfg.use_graphkan and first_out is not None:
            save_adjacency_heatmap(
                first_out["adj"],
                str(explain_dir / "adaptive_adjacency.png"),
                title=f"Adaptive Adjacency - {condition_name} - {ablation_name(cfg)}",
            )
            explain_graph_propagation(
                model,
                str(explain_dir / "graph_propagation"),
                topk=5,
                num_points=200,
            )

        if cfg.regressor_type.lower() == "kan":
            explain_output_mapping(
                model,
                str(explain_dir / "output_mapping"),
                topk=8,
                num_points=200,
            )

    plot_dir = result_root / ablation_name(cfg) / condition_name / "rul_plots"
    final_df = evaluate_per_bearing(
        model,
        test_loader,
        cfg.device,
        plot_dir=str(plot_dir),
        plot_title_prefix=condition_name,
    )
    final_df.insert(0, "Ablation", ablation_name(cfg))

    final_overall = {
        "RMSE": float(final_df["RMSE"].mean()),
        "MAE": float(final_df["MAE"].mean()),
        "EAS": float(final_df["EAS"].mean()),
    }

    csv_path = result_root / ablation_name(cfg) / condition_name / f"phm2012_{condition_name.lower()}_per_bearing.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(csv_path, index=False)

    print(f"\n========== Final Overall ({condition_name}) ==========")
    print(final_overall)
    print(f"Saved per-bearing results to {csv_path}")
    return final_overall, final_df


def run_all_conditions(cfg: PHM2012Config, explain: bool = False):
    set_seed(cfg.seed)
    result_root = cfg.make_run_dir("phm2012")

    all_condition_rows = []
    all_bearing_dfs = []

    print(f"\nRunning setting: {ablation_name(cfg)}")
    print(f"Output directory: {result_root}")

    for condition_name in PHM2012_CONDITIONS.keys():
        cond_overall, cond_df = run_condition(cfg, condition_name, result_root, explain=explain)

        row = {"Ablation": ablation_name(cfg), "Condition": condition_name}
        row.update(cond_overall)
        all_condition_rows.append(row)

        cond_df = cond_df.copy()
        cond_df.insert(1, "Condition", condition_name)
        all_bearing_dfs.append(cond_df)

    condition_summary_df = pd.DataFrame(all_condition_rows)
    all_bearing_df = pd.concat(all_bearing_dfs, axis=0, ignore_index=True)

    save_root = result_root / ablation_name(cfg)
    save_root.mkdir(parents=True, exist_ok=True)
    condition_summary_path = save_root / "phm2012_condition_summary.csv"
    all_bearing_path = save_root / "phm2012_all_conditions_per_bearing.csv"

    condition_summary_df.to_csv(condition_summary_path, index=False)
    all_bearing_df.to_csv(all_bearing_path, index=False)

    overall_summary = {
        "RMSE": float(all_bearing_df["RMSE"].mean()),
        "MAE": float(all_bearing_df["MAE"].mean()),
        "EAS": float(all_bearing_df["EAS"].mean()),
    }

    print("\n========== Condition Summary ==========")
    print(condition_summary_df)
    print("\n========== All Test Bearings Summary ==========")
    print(overall_summary)
    print(f"\nSaved condition summary to {condition_summary_path}")
    print(f"Saved all per-bearing results to {all_bearing_path}")
    return overall_summary, condition_summary_df, all_bearing_df


def parse_args():
    parser = argparse.ArgumentParser(description="Train Bi-KAGN on PHM2012/PRONOSTIA.")
    parser.add_argument("--data-root", default="./ieee-phm-2012-data-challenge-dataset-master")
    parser.add_argument("--train-folder", default="Learning_set")
    parser.add_argument("--test-folder", default="Full_Test_Set")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--regressor", choices=["kan", "mlp", "multkan"], default="kan")
    parser.add_argument("--use-gat", action="store_true", help="Use GAT baseline instead of GraphKAN.")
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--explain", action="store_true", help="Save KAN function-response visualizations.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PHM2012Config(
        data_root=args.data_root,
        train_folder=args.train_folder,
        test_folder=args.test_folder,
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
    run_all_conditions(cfg, explain=args.explain)


if __name__ == "__main__":
    main()
