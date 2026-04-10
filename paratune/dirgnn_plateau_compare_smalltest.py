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

THETA_LIBRARY: dict[str, tuple[float, float, float, float, float, float]] = {
    "balanced_custom": (-1.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    "baseline": (-0.5, 1.0, 0.8, 1.0, -0.6, 1.0),
    "symmetric_moderate": (-0.5, 1.0, 1.0, 0.8, 0.8, 1.0),
    "in_dominant": (-0.5, 1.8, 0.4, 1.6, 0.1, 1.0),
    "out_dominant": (-0.5, 0.4, 1.8, 0.1, 1.6, 1.0),
    "asym_signflip": (-0.5, 1.8, 0.2, 1.8, -1.4, 1.0),
    "weak_directional": (-0.5, 0.6, 0.5, 0.3, -0.2, 1.0),
}


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


def one_vs_rest_fpr_fnr(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> tuple[float, float]:
    pos = (y_true == int(cls))
    pred_pos = (y_pred == int(cls))
    tp = int(np.sum(pos & pred_pos))
    fn = int(np.sum(pos & (~pred_pos)))
    fp = int(np.sum((~pos) & pred_pos))
    tn = int(np.sum((~pos) & (~pred_pos)))
    fnr = (fn / (tp + fn)) if (tp + fn) > 0 else float("nan")
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    return float(fpr), float(fnr)


def macro_fpr_fnr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    fprs: list[float] = []
    fnrs: list[float] = []
    for cls in range(8):
        fpr, fnr = one_vs_rest_fpr_fnr(y_true, y_pred, cls)
        fprs.append(float(fpr))
        fnrs.append(float(fnr))
    return float(np.nanmean(np.asarray(fprs))), float(np.nanmean(np.asarray(fnrs)))


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
        f"plateau compare started theta={args.theta_name} n={args.n} num_runs={args.num_runs} "
        f"val_frac={args.val_frac} plateau_patience={args.plateau_patience}",
        flush=True,
    )
    t0 = time.time()
    try:
        gen_data.THETA_D_DIR = theta
        for run_idx in range(int(args.num_runs)):
            seed = int(args.seed_start) + run_idx
            draw = gen_data.sample_data_spillover(
                sample_size=int(args.n),
                seed=seed,
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
            val_mask = make_stratified_holdout(state, val_frac=float(args.val_frac), seed=seed + 707)
            train_mask = np.logical_not(val_mask)
            treated_prop = float(np.mean(np.asarray(draw["D"], dtype=float)))

            for variant, use_plateau in (("dirgnn_base", False), ("dirgnn_plateau", True)):
                y_hat = np.asarray(
                    GNN_reg_dir(
                        Y=y,
                        X=x_aug,
                        A=a,
                        num_layers=int(args.num_layers),
                        output_dim=int(args.output_dim),
                        sample=train_mask,
                        seed=seed,
                        use_gpu=bool(int(args.use_gpu)),
                        use_plateau=bool(use_plateau),
                        plateau_factor=float(args.plateau_factor),
                        plateau_patience=int(args.plateau_patience),
                        plateau_min_lr=float(args.plateau_min_lr),
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
                        sample=train_mask,
                        seed=seed,
                        use_gpu=bool(int(args.use_gpu)),
                        use_plateau=bool(use_plateau),
                        plateau_factor=float(args.plateau_factor),
                        plateau_patience=int(args.plateau_patience),
                        plateau_min_lr=float(args.plateau_min_lr),
                    ),
                    dtype=float,
                )
                probs_val = probs[val_mask]
                state_val = state[val_mask]
                eps = 1e-12
                gps_ce = float(-np.mean(np.log(probs_val[np.arange(state_val.size), state_val] + eps)))
                pred_val = np.asarray(np.argmax(probs_val, axis=1), dtype=int)
                macro_fpr, macro_fnr = macro_fpr_fnr(state_val, pred_val)

                run_rows.append(
                    {
                        "theta_name": str(args.theta_name),
                        "theta_d_dir": str(theta),
                        "run": int(run_idx + 1),
                        "seed": int(seed),
                        "n": int(args.n),
                        "val_frac": float(args.val_frac),
                        "n_train": int(np.sum(train_mask)),
                        "n_val": int(np.sum(val_mask)),
                        "treated_proportion": float(treated_prop),
                        "variant": variant,
                        "use_plateau": int(use_plateau),
                        "plateau_factor": float(args.plateau_factor),
                        "plateau_patience": int(args.plateau_patience),
                        "plateau_min_lr": float(args.plateau_min_lr),
                        "outcome_mse": float(outcome_mse),
                        "gps_ce": float(gps_ce),
                        "gps_macro_fpr": float(macro_fpr),
                        "gps_macro_fnr": float(macro_fnr),
                    }
                )
            print(
                f"completed run {run_idx + 1}/{args.num_runs} seed={seed} treated_prop={treated_prop:.4f}",
                flush=True,
            )
    finally:
        gen_data.THETA_D_DIR = original_theta

    for variant in ("dirgnn_base", "dirgnn_plateau"):
        rs = [r for r in run_rows if r["variant"] == variant]
        outcome = np.asarray([float(r["outcome_mse"]) for r in rs], dtype=float)
        ce = np.asarray([float(r["gps_ce"]) for r in rs], dtype=float)
        fpr = np.asarray([float(r["gps_macro_fpr"]) for r in rs], dtype=float)
        fnr = np.asarray([float(r["gps_macro_fnr"]) for r in rs], dtype=float)
        summary_rows.append(
            {
                "theta_name": str(args.theta_name),
                "theta_d_dir": str(theta),
                "n": int(args.n),
                "num_runs": int(args.num_runs),
                "variant": variant,
                "mean_outcome_mse": float(np.mean(outcome)),
                "std_outcome_mse": float(np.std(outcome, ddof=1)) if outcome.size > 1 else 0.0,
                "mean_gps_ce": float(np.mean(ce)),
                "std_gps_ce": float(np.std(ce, ddof=1)) if ce.size > 1 else 0.0,
                "mean_gps_macro_fpr": float(np.mean(fpr)),
                "std_gps_macro_fpr": float(np.std(fpr, ddof=1)) if fpr.size > 1 else 0.0,
                "mean_gps_macro_fnr": float(np.mean(fnr)),
                "std_gps_macro_fnr": float(np.std(fnr, ddof=1)) if fnr.size > 1 else 0.0,
            }
        )

    base = next(r for r in summary_rows if r["variant"] == "dirgnn_base")
    plat = next(r for r in summary_rows if r["variant"] == "dirgnn_plateau")
    summary_rows.append(
        {
            "theta_name": str(args.theta_name),
            "theta_d_dir": str(theta),
            "n": int(args.n),
            "num_runs": int(args.num_runs),
            "variant": "delta_base_minus_plateau",
            "mean_outcome_mse": float(base["mean_outcome_mse"]) - float(plat["mean_outcome_mse"]),
            "std_outcome_mse": float("nan"),
            "mean_gps_ce": float(base["mean_gps_ce"]) - float(plat["mean_gps_ce"]),
            "std_gps_ce": float("nan"),
            "mean_gps_macro_fpr": float(base["mean_gps_macro_fpr"]) - float(plat["mean_gps_macro_fpr"]),
            "std_gps_macro_fpr": float("nan"),
            "mean_gps_macro_fnr": float(base["mean_gps_macro_fnr"]) - float(plat["mean_gps_macro_fnr"]),
            "std_gps_macro_fnr": float("nan"),
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
    elapsed = time.time() - t0
    print(f"wrote run-level csv: {run_csv}", flush=True)
    print(f"wrote summary csv: {summary_csv}", flush=True)
    print(f"done in {elapsed:.1f}s", flush=True)
    return run_csv, summary_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare DirGNN base vs DirGNN+ReduceLROnPlateau for outcome MSE and GPS metrics."
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
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_patience", type=int, default=8)
    parser.add_argument("--plateau_min_lr", type=float, default=1e-5)
    parser.add_argument("--run_csv", type=str, default="paratune/dirgnn_plateau_compare_runs.csv")
    parser.add_argument("--summary_csv", type=str, default="paratune/dirgnn_plateau_compare_summary.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
