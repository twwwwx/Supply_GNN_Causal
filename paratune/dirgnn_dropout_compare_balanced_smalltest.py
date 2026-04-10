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
from src.GNN import GNN_reg_dir, GNN_reg_dir_multiclass

THETA_BALANCED = (-1.0, 1.0, 0.0, 1.0, 0.0, 1.0)


def parse_float_list(spec: str) -> list[float]:
    vals = [s.strip() for s in str(spec).split(",") if s.strip()]
    if not vals:
        raise ValueError("dropouts list is empty.")
    return [float(v) for v in vals]


def make_stratified_holdout(labels: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    y = np.asarray(labels, dtype=int).reshape(-1)
    n = int(y.size)
    val_mask = np.zeros(n, dtype=bool)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        k = max(1, int(np.floor(float(val_frac) * idx.size)))
        val_mask[idx[:k]] = True
    if not np.any(val_mask):
        val_mask[rng.integers(0, n)] = True
    return val_mask


def macro_fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    fnrs: list[float] = []
    for cls in range(8):
        pos = (y_true == cls)
        tp = int(np.sum(pos & (y_pred == cls)))
        fn = int(np.sum(pos & (y_pred != cls)))
        if (tp + fn) == 0:
            fnrs.append(float("nan"))
        else:
            fnrs.append(float(fn / (tp + fn)))
    return float(np.nanmean(np.asarray(fnrs, dtype=float)))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    dropouts = parse_float_list(args.dropouts)
    original_theta = tuple(gen_data.THETA_D_DIR)
    run_rows: list[dict] = []
    summary_rows: list[dict] = []

    print(
        f"dirgnn dropout compare started theta={THETA_BALANCED} "
        f"n={args.n} num_runs={args.num_runs} dropouts={dropouts}",
        flush=True,
    )
    start = time.time()
    try:
        gen_data.THETA_D_DIR = THETA_BALANCED
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
            y = np.asarray(draw["Y"], dtype=float).reshape(-1)
            x = np.asarray(draw["node_features"], dtype=float)
            t = np.asarray(draw["T"], dtype=float)
            x_aug = np.concatenate([x, t], axis=1)
            a = np.asarray(draw["adjacency"])
            state = np.asarray(draw["state_index"], dtype=int).reshape(-1)
            val_mask = make_stratified_holdout(state, val_frac=float(args.val_frac), seed=seed + 701)
            train_mask = np.logical_not(val_mask)
            treated_prop = float(np.mean(np.asarray(draw["D"], dtype=float)))

            for dropout in dropouts:
                y_hat = np.asarray(
                    GNN_reg_dir(
                        Y=y,
                        X=x_aug,
                        A=a,
                        num_layers=int(args.num_layers),
                        output_dim=int(args.output_dim),
                        dropout=float(dropout),
                        sample=train_mask,
                        seed=int(seed),
                        use_gpu=bool(int(args.use_gpu)),
                        use_plateau=False,
                    ),
                    dtype=float,
                ).reshape(-1)
                err = y_hat[val_mask] - y[val_mask]
                outcome_mse = float(np.mean(err * err))

                probs = np.asarray(
                    GNN_reg_dir_multiclass(
                        labels=state,
                        X=x,
                        A=a,
                        num_classes=8,
                        num_layers=int(args.num_layers),
                        output_dim=int(args.output_dim),
                        dropout=float(dropout),
                        sample=train_mask,
                        seed=int(seed),
                        use_gpu=bool(int(args.use_gpu)),
                        use_plateau=False,
                    ),
                    dtype=float,
                )
                pred_val = np.asarray(np.argmax(probs[val_mask], axis=1), dtype=int)
                state_val = state[val_mask]
                gps_macro_fnr = float(macro_fnr(state_val, pred_val))

                run_rows.append(
                    {
                        "theta_d_dir": str(THETA_BALANCED),
                        "run": int(run_idx + 1),
                        "seed": int(seed),
                        "n": int(args.n),
                        "val_frac": float(args.val_frac),
                        "n_train": int(np.sum(train_mask)),
                        "n_val": int(np.sum(val_mask)),
                        "treated_proportion": float(treated_prop),
                        "dropout": float(dropout),
                        "outcome_mse": float(outcome_mse),
                        "gps_macro_fnr": float(gps_macro_fnr),
                    }
                )
            print(
                f"completed run {run_idx + 1}/{args.num_runs} seed={seed} treated_prop={treated_prop:.4f}",
                flush=True,
            )
    finally:
        gen_data.THETA_D_DIR = original_theta

    for dropout in dropouts:
        rs = [r for r in run_rows if float(r["dropout"]) == float(dropout)]
        mse = np.asarray([float(r["outcome_mse"]) for r in rs], dtype=float)
        fnr = np.asarray([float(r["gps_macro_fnr"]) for r in rs], dtype=float)
        summary_rows.append(
            {
                "theta_d_dir": str(THETA_BALANCED),
                "n": int(args.n),
                "num_runs": int(args.num_runs),
                "dropout": float(dropout),
                "mean_outcome_mse": float(np.mean(mse)),
                "std_outcome_mse": float(np.std(mse, ddof=1)) if mse.size > 1 else 0.0,
                "mean_gps_macro_fnr": float(np.mean(fnr)),
                "std_gps_macro_fnr": float(np.std(fnr, ddof=1)) if fnr.size > 1 else 0.0,
            }
        )

    summary_rows.sort(key=lambda r: (float(r["mean_outcome_mse"]), float(r["mean_gps_macro_fnr"])))
    for rank, row in enumerate(summary_rows, start=1):
        row["rank_joint_low"] = int(rank)

    run_csv = Path(args.run_csv)
    if not run_csv.is_absolute():
        run_csv = ROOT / run_csv
    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_absolute():
        summary_csv = ROOT / summary_csv

    write_csv(run_csv, run_rows)
    write_csv(summary_csv, summary_rows)
    elapsed = time.time() - start
    print(f"wrote run-level csv: {run_csv}", flush=True)
    print(f"wrote summary csv: {summary_csv}", flush=True)
    print(f"done in {elapsed:.1f}s", flush=True)
    return run_csv, summary_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare DirGNN dropout values on balanced design for outcome MSE and GPS macro FNR."
    )
    parser.add_argument("--dropouts", type=str, default="0,0.01,0.1")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--seed_start", type=int, default=123)
    parser.add_argument("--val_frac", type=float, default=0.3)
    parser.add_argument("--p_bidirected", type=float, default=0.05)
    parser.add_argument("--tau_dir_true", type=float, default=-2.0)
    parser.add_argument("--tau_in_true", type=float, default=1.0)
    parser.add_argument("--tau_out_true", type=float, default=1.0)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--run_csv", type=str, default="paratune/dirgnn_dropout_compare_runs_balanced.csv")
    parser.add_argument("--summary_csv", type=str, default="paratune/dirgnn_dropout_compare_summary_balanced.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

