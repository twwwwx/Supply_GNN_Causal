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
from src.GNN import GNN_reg, GNN_reg_dir

THETA_LIBRARY: dict[str, tuple[float, float, float, float, float, float]] = {
    "balanced_custom": (-1.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    "baseline": (-0.5, 1.0, 0.8, 1.0, -0.6, 1.0),
    "symmetric_moderate": (-0.5, 1.0, 1.0, 0.8, 0.8, 1.0),
    "in_dominant": (-0.5, 1.8, 0.4, 1.6, 0.1, 1.0),
    "out_dominant": (-0.5, 0.4, 1.8, 0.1, 1.6, 1.0),
    "asym_signflip": (-0.5, 1.8, 0.2, 1.8, -1.4, 1.0),
    "weak_directional": (-0.5, 0.6, 0.5, 0.3, -0.2, 1.0),
}


def make_stratified_holdout(
    labels: np.ndarray,
    val_frac: float,
    seed: int,
) -> np.ndarray:
    if not (0.0 < float(val_frac) < 1.0):
        raise ValueError("val_frac must be in (0, 1).")
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    theta = THETA_LIBRARY[str(args.theta_name)]
    original_theta = tuple(gen_data.THETA_D_DIR)
    run_rows: list[dict] = []
    summary_rows: list[dict] = []

    print(
        f"outcome mse small test started theta={args.theta_name} "
        f"n={args.n} num_runs={args.num_runs} val_frac={args.val_frac} output_dim={args.output_dim}",
        flush=True,
    )
    start = time.time()
    try:
        gen_data.THETA_D_DIR = theta
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
            val_mask = make_stratified_holdout(state, val_frac=float(args.val_frac), seed=seed + 77)
            train_mask = np.logical_not(val_mask)
            treated_prop = float(np.mean(np.asarray(draw["D"], dtype=float)))

            for model in ("gnn", "dirgnn"):
                if model == "gnn":
                    y_hat = np.asarray(
                        GNN_reg(
                            Y=y,
                            X=x_aug,
                            A=a,
                            num_layers=int(args.num_layers),
                            output_dim=int(args.output_dim),
                            sample=train_mask,
                            seed=int(seed),
                            use_gpu=bool(int(args.use_gpu)),
                        ),
                        dtype=float,
                    ).reshape(-1)
                else:
                    y_hat = np.asarray(
                        GNN_reg_dir(
                            Y=y,
                            X=x_aug,
                            A=a,
                            num_layers=int(args.num_layers),
                            output_dim=int(args.output_dim),
                            sample=train_mask,
                            seed=int(seed),
                            use_gpu=bool(int(args.use_gpu)),
                        ),
                        dtype=float,
                    ).reshape(-1)

                err = y_hat[val_mask] - y[val_mask]
                mse = float(np.mean(err * err))
                rmse = float(np.sqrt(mse))
                run_rows.append(
                    {
                        "theta_name": str(args.theta_name),
                        "theta_d_dir": str(theta),
                        "run": int(run_idx + 1),
                        "seed": int(seed),
                        "n": int(args.n),
                        "treated_proportion": float(treated_prop),
                        "model": model,
                        "val_frac": float(args.val_frac),
                        "n_train": int(np.sum(train_mask)),
                        "n_val": int(np.sum(val_mask)),
                        "mse": mse,
                        "rmse": rmse,
                    }
                )
            print(
                f"completed run {run_idx + 1}/{args.num_runs} seed={seed} treated_prop={treated_prop:.4f}",
                flush=True,
            )
    finally:
        gen_data.THETA_D_DIR = original_theta

    for model in ("gnn", "dirgnn"):
        rows_m = [r for r in run_rows if r["model"] == model]
        m = np.asarray([float(r["mse"]) for r in rows_m], dtype=float)
        r = np.asarray([float(r["rmse"]) for r in rows_m], dtype=float)
        summary_rows.append(
            {
                "theta_name": str(args.theta_name),
                "theta_d_dir": str(theta),
                "n": int(args.n),
                "num_runs": int(args.num_runs),
                "model": model,
                "mean_mse": float(np.mean(m)),
                "std_mse": float(np.std(m, ddof=1)) if m.size > 1 else 0.0,
                "mean_rmse": float(np.mean(r)),
                "std_rmse": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
            }
        )

    g = next(row for row in summary_rows if row["model"] == "gnn")
    d = next(row for row in summary_rows if row["model"] == "dirgnn")
    summary_rows.append(
        {
            "theta_name": str(args.theta_name),
            "theta_d_dir": str(theta),
            "n": int(args.n),
            "num_runs": int(args.num_runs),
            "model": "delta_gnn_minus_dirgnn",
            "mean_mse": float(g["mean_mse"]) - float(d["mean_mse"]),
            "std_mse": float("nan"),
            "mean_rmse": float(g["mean_rmse"]) - float(d["mean_rmse"]),
            "std_rmse": float("nan"),
        }
    )

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
        description="Small holdout test for outcome regression MSE under a chosen THETA_D_DIR."
    )
    parser.add_argument("--theta_name", type=str, default="out_dominant", choices=tuple(sorted(THETA_LIBRARY.keys())))
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
    parser.add_argument("--run_csv", type=str, default="paratune/outcome_mse_runs_smalltest.csv")
    parser.add_argument("--summary_csv", type=str, default="paratune/outcome_mse_summary_smalltest.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
