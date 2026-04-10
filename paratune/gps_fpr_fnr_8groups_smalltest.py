#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.gen_data as gen_data
from src.GNN import GNN_reg_dir_multiclass, GNN_reg_multiclass

THETA_LIBRARY: dict[str, tuple[float, float, float, float, float, float]] = {
    "balanced_custom": (-1.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    "baseline": (-0.5, 1.0, 0.8, 1.0, -0.6, 1.0),
    "symmetric_moderate": (-0.5, 1.0, 1.0, 0.8, 0.8, 1.0),
    "in_dominant": (-0.5, 1.8, 0.4, 1.6, 0.1, 1.0),
    "out_dominant": (-0.5, 0.4, 1.8, 0.1, 1.6, 1.0),
    "asym_signflip": (-0.5, 1.8, 0.2, 1.8, -1.4, 1.0),
    "weak_directional": (-0.5, 0.6, 0.5, 0.3, -0.2, 1.0),
}


def idx_to_group_label(idx: int) -> str:
    d = (int(idx) >> 2) & 1
    rin = (int(idx) >> 1) & 1
    rout = int(idx) & 1
    return f"D{d}_rin{rin}_rout{rout}"


def _fpr_fnr_one_vs_rest(y_true: np.ndarray, y_pred: np.ndarray, class_idx: int) -> dict[str, float]:
    positive = (y_true == int(class_idx))
    pred_pos = (y_pred == int(class_idx))

    tp = int(np.sum(positive & pred_pos))
    fn = int(np.sum(positive & (~pred_pos)))
    fp = int(np.sum((~positive) & pred_pos))
    tn = int(np.sum((~positive) & (~pred_pos)))

    fnr = (fn / (tp + fn)) if (tp + fn) > 0 else float("nan")
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    return {
        "tp": float(tp),
        "fn": float(fn),
        "fp": float(fp),
        "tn": float(tn),
        "fnr": float(fnr),
        "fpr": float(fpr),
        "support_pos": float(tp + fn),
        "support_neg": float(fp + tn),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _predict_state_probs(
    model: str,
    labels: np.ndarray,
    x: np.ndarray,
    a: np.ndarray,
    num_layers: int,
    output_dim: int,
    seed: int,
    use_gpu: bool,
) -> np.ndarray:
    if model == "gnn":
        return np.asarray(
            GNN_reg_multiclass(
                labels=labels,
                X=x,
                A=a,
                num_classes=8,
                num_layers=int(num_layers),
                output_dim=int(output_dim),
                seed=int(seed),
                use_gpu=bool(use_gpu),
            ),
            dtype=float,
        )
    if model == "dirgnn":
        return np.asarray(
            GNN_reg_dir_multiclass(
                labels=labels,
                X=x,
                A=a,
                num_classes=8,
                num_layers=int(num_layers),
                output_dim=int(output_dim),
                seed=int(seed),
                use_gpu=bool(use_gpu),
            ),
            dtype=float,
        )
    raise ValueError(f"Unknown model: {model}")


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    theta = THETA_LIBRARY[args.theta_name]
    original_theta = tuple(gen_data.THETA_D_DIR)
    run_rows: list[dict] = []
    summary_rows: list[dict] = []
    started = time.time()

    try:
        gen_data.THETA_D_DIR = theta
        print(
            f"gps fpr/fnr small test started: theta_name={args.theta_name} theta={theta} "
            f"n={args.n} num_runs={args.num_runs} output_dim={args.output_dim}",
            flush=True,
        )
        for run_idx in range(int(args.num_runs)):
            seed = int(args.seed_start) + run_idx
            draw = gen_data.sample_data_spillover(
                sample_size=int(args.n),
                seed=int(seed),
                graph_model="rgg",
                p_bidirected=float(args.p_bidirected),
                tau_dir=float(args.tau_dir_true),
                tau_in=float(args.tau_in_true),
                tau_out=float(args.tau_out_true),
            )
            state_true = np.asarray(draw["state_index"], dtype=int).reshape(-1)
            x = np.asarray(draw["node_features"], dtype=float)
            a = np.asarray(draw["adjacency"])
            treated_prop = float(np.mean(np.asarray(draw["D"], dtype=float)))

            for model in ("gnn", "dirgnn"):
                probs = _predict_state_probs(
                    model=model,
                    labels=state_true,
                    x=x,
                    a=a,
                    num_layers=int(args.num_layers),
                    output_dim=int(args.output_dim),
                    seed=int(seed),
                    use_gpu=bool(int(args.use_gpu)),
                )
                pred = np.asarray(np.argmax(probs, axis=1), dtype=int)
                for class_idx in range(8):
                    stats = _fpr_fnr_one_vs_rest(state_true, pred, class_idx)
                    row = {
                        "theta_name": args.theta_name,
                        "theta_d_dir": str(theta),
                        "run": int(run_idx + 1),
                        "seed": int(seed),
                        "n": int(args.n),
                        "treated_proportion": float(treated_prop),
                        "model": model,
                        "state_idx": int(class_idx),
                        "group": idx_to_group_label(class_idx),
                        "state_prevalence": float(np.mean(state_true == class_idx)),
                        "fpr": float(stats["fpr"]),
                        "fnr": float(stats["fnr"]),
                        "tp": int(stats["tp"]),
                        "fn": int(stats["fn"]),
                        "fp": int(stats["fp"]),
                        "tn": int(stats["tn"]),
                    }
                    run_rows.append(row)

            print(
                f"completed run {run_idx + 1}/{args.num_runs} seed={seed} treated_prop={treated_prop:.4f}",
                flush=True,
            )
    finally:
        gen_data.THETA_D_DIR = original_theta

    for model in ("gnn", "dirgnn"):
        for class_idx in range(8):
            subset = [r for r in run_rows if r["model"] == model and int(r["state_idx"]) == class_idx]
            fpr_vals = np.asarray([float(r["fpr"]) for r in subset], dtype=float)
            fnr_vals = np.asarray([float(r["fnr"]) for r in subset], dtype=float)
            prev_vals = np.asarray([float(r["state_prevalence"]) for r in subset], dtype=float)
            summary_rows.append(
                {
                    "theta_name": args.theta_name,
                    "theta_d_dir": str(theta),
                    "n": int(args.n),
                    "num_runs": int(args.num_runs),
                    "model": model,
                    "state_idx": int(class_idx),
                    "group": idx_to_group_label(class_idx),
                    "mean_state_prevalence": float(np.mean(prev_vals)),
                    "mean_fpr": float(np.nanmean(fpr_vals)),
                    "std_fpr": float(np.nanstd(fpr_vals, ddof=1)) if fpr_vals.size > 1 else 0.0,
                    "mean_fnr": float(np.nanmean(fnr_vals)),
                    "std_fnr": float(np.nanstd(fnr_vals, ddof=1)) if fnr_vals.size > 1 else 0.0,
                }
            )

    for model in ("gnn", "dirgnn"):
        subset = [r for r in summary_rows if r["model"] == model]
        fpr_vals = np.asarray([float(r["mean_fpr"]) for r in subset], dtype=float)
        fnr_vals = np.asarray([float(r["mean_fnr"]) for r in subset], dtype=float)
        summary_rows.append(
            {
                "theta_name": args.theta_name,
                "theta_d_dir": str(theta),
                "n": int(args.n),
                "num_runs": int(args.num_runs),
                "model": model,
                "state_idx": -1,
                "group": "macro_avg",
                "mean_state_prevalence": float(np.mean(np.asarray([float(r["mean_state_prevalence"]) for r in subset]))),
                "mean_fpr": float(np.nanmean(fpr_vals)),
                "std_fpr": float(np.nanstd(fpr_vals, ddof=1)) if fpr_vals.size > 1 else 0.0,
                "mean_fnr": float(np.nanmean(fnr_vals)),
                "std_fnr": float(np.nanstd(fnr_vals, ddof=1)) if fnr_vals.size > 1 else 0.0,
            }
        )

    run_csv = Path(args.run_csv)
    if not run_csv.is_absolute():
        run_csv = ROOT / run_csv
    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_absolute():
        summary_csv = ROOT / summary_csv

    _write_csv(run_csv, run_rows)
    _write_csv(summary_csv, summary_rows)

    elapsed = time.time() - started
    print(f"wrote run-level csv: {run_csv}", flush=True)
    print(f"wrote summary csv: {summary_csv}", flush=True)
    print(f"done in {elapsed:.1f}s", flush=True)
    return run_csv, summary_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Small test: compare GNN vs DirGNN generalized propensity prediction with "
            "one-vs-rest FPR/FNR across all 8 treatment-exposure groups."
        )
    )
    parser.add_argument("--theta_name", type=str, default="out_dominant", choices=tuple(sorted(THETA_LIBRARY.keys())))
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--seed_start", type=int, default=123)
    parser.add_argument("--p_bidirected", type=float, default=0.05)
    parser.add_argument("--tau_dir_true", type=float, default=-2.0)
    parser.add_argument("--tau_in_true", type=float, default=1.0)
    parser.add_argument("--tau_out_true", type=float, default=1.0)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--run_csv", type=str, default="paratune/gps_fpr_fnr_8groups_runs.csv")
    parser.add_argument("--summary_csv", type=str, default="paratune/gps_fpr_fnr_8groups_summary.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
