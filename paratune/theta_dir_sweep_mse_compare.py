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
from src.ate import tau_vector_and_se_from_gnn

ESTIMANDS = ("tau_dir", "tau_in", "tau_out", "tau_tot")

THETA_CANDIDATES: list[tuple[str, tuple[float, float, float, float, float, float]]] = [
    ("baseline", (-0.5, 1.0, 0.8, 1.0, -0.6, 1.0)),
    ("symmetric_moderate", (-0.5, 1.0, 1.0, 0.8, 0.8, 1.0)),
    ("in_dominant", (-0.5, 1.8, 0.4, 1.6, 0.1, 1.0)),
    ("out_dominant", (-0.5, 0.4, 1.8, 0.1, 1.6, 1.0)),
    ("asym_signflip", (-0.5, 1.8, 0.2, 1.8, -1.4, 1.0)),
    ("weak_directional", (-0.5, 0.6, 0.5, 0.3, -0.2, 1.0)),
]


def smoke_check_treated_proportion(
    *,
    sample_size: int,
    graph_model: str,
    p_bidirected: float,
    theta_idx: int,
    smoke_trials: int,
    smoke_seed_start: int,
    max_treated_prop: float,
) -> tuple[bool, float, float]:
    props: list[float] = []
    for offset in range(int(smoke_trials)):
        smoke_seed = int(smoke_seed_start) + (1000 * int(theta_idx)) + offset
        draw = gen_data.sample_data_dir(
            sample_size=int(sample_size),
            seed=int(smoke_seed),
            graph_model=graph_model,
            p_bidirected=float(p_bidirected),
        )
        treated_prop = float(np.mean(np.asarray(draw["D"], dtype=float)))
        props.append(treated_prop)
    max_prop = float(np.max(np.asarray(props, dtype=float)))
    mean_prop = float(np.mean(np.asarray(props, dtype=float)))
    passed = bool(max_prop <= float(max_treated_prop))
    return passed, max_prop, mean_prop


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    use_gpu = bool(int(args.use_gpu))
    original_theta = tuple(gen_data.THETA_D_DIR)
    summary_rows: list[dict] = []
    compare_rows: list[dict] = []

    print(
        f"theta sweep started n={args.n} num_runs={args.num_runs} output_dim={args.output_dim} "
        f"clip={args.clip} num_layers={args.num_layers} use_gpu={int(use_gpu)}",
        flush=True,
    )

    try:
        for theta_idx, (theta_name, theta_tuple) in enumerate(THETA_CANDIDATES, start=1):
            theta_start = time.time()
            gen_data.THETA_D_DIR = theta_tuple
            print(
                f"[{theta_idx}/{len(THETA_CANDIDATES)}] theta={theta_name} THETA_D_DIR={theta_tuple}",
                flush=True,
            )
            smoke_passed, smoke_max_prop, smoke_mean_prop = smoke_check_treated_proportion(
                sample_size=int(args.n),
                graph_model="rgg",
                p_bidirected=float(args.p_bidirected),
                theta_idx=int(theta_idx),
                smoke_trials=int(args.smoke_trials),
                smoke_seed_start=int(args.smoke_seed_start),
                max_treated_prop=float(args.max_treated_prop),
            )
            print(
                f"theta={theta_name} smoke_check treated_prop_max={smoke_max_prop:.6f} "
                f"treated_prop_mean={smoke_mean_prop:.6f} pass={int(smoke_passed)} "
                f"(threshold={float(args.max_treated_prop):.3f})",
                flush=True,
            )
            if not smoke_passed:
                msg = (
                    f"theta={theta_name} skipped: smoke check failed "
                    f"(treated_prop_max={smoke_max_prop:.6f} > {float(args.max_treated_prop):.3f})."
                )
                if args.smoke_fail_action == "error":
                    raise RuntimeError(msg)
                print(msg, flush=True)
                continue

            sq_err_by_model: dict[str, dict[str, list[float]]] = {
                "gnn": {k: [] for k in ESTIMANDS},
                "dirgnn": {k: [] for k in ESTIMANDS},
            }

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
                true_taus = draw["true_taus"]

                for model_name, directed in (("gnn", False), ("dirgnn", True)):
                    fit = tau_vector_and_se_from_gnn(
                        draw,
                        feature_key="node_features",
                        clip=float(args.clip),
                        num_layers=int(args.num_layers),
                        output_dim=int(args.output_dim),
                        seed=seed,
                        directed=directed,
                        use_gpu=use_gpu,
                        variance_type="iid",  # variance choice does not affect tau_hat MSE
                    )
                    for estimand in ESTIMANDS:
                        err = float(fit["tau_hat"][estimand]) - float(true_taus[estimand])
                        sq_err_by_model[model_name][estimand].append(err * err)
                if (run_idx + 1) % int(args.log_every_run) == 0 or (run_idx + 1) == int(args.num_runs):
                    print(
                        f"theta={theta_name} completed_runs={run_idx + 1}/{args.num_runs}",
                        flush=True,
                    )

            theta_elapsed = time.time() - theta_start
            model_rows: dict[str, dict] = {}
            for model_name in ("gnn", "dirgnn"):
                row: dict[str, float | int | str] = {
                    "theta_name": theta_name,
                    "theta_d_dir": str(theta_tuple),
                    "model": model_name,
                    "n": int(args.n),
                    "num_runs": int(args.num_runs),
                    "clip": float(args.clip),
                    "num_layers": int(args.num_layers),
                    "output_dim": int(args.output_dim),
                    "smoke_trials": int(args.smoke_trials),
                    "smoke_max_treated_prop": float(smoke_max_prop),
                    "smoke_mean_treated_prop": float(smoke_mean_prop),
                    "runtime_sec_theta_block": float(theta_elapsed),
                }
                mse_all = []
                for estimand in ESTIMANDS:
                    mse_e = float(np.mean(np.asarray(sq_err_by_model[model_name][estimand], dtype=float)))
                    row[f"mse_{estimand}"] = mse_e
                    mse_all.append(mse_e)
                row["mean_mse_all"] = float(np.mean(np.asarray(mse_all, dtype=float)))
                summary_rows.append(row)
                model_rows[model_name] = row

            g = model_rows["gnn"]
            d = model_rows["dirgnn"]
            cmp: dict[str, float | int | str] = {
                "theta_name": theta_name,
                "theta_d_dir": str(theta_tuple),
                "n": int(args.n),
                "num_runs": int(args.num_runs),
                "smoke_trials": int(args.smoke_trials),
                "smoke_max_treated_prop": float(smoke_max_prop),
                "smoke_mean_treated_prop": float(smoke_mean_prop),
                "gnn_mean_mse_all": float(g["mean_mse_all"]),
                "dirgnn_mean_mse_all": float(d["mean_mse_all"]),
                "delta_mean_mse_all": float(g["mean_mse_all"]) - float(d["mean_mse_all"]),
                "ratio_mean_mse_all": (
                    float(g["mean_mse_all"]) / float(d["mean_mse_all"])
                    if float(d["mean_mse_all"]) > 0
                    else float("inf")
                ),
            }
            for estimand in ESTIMANDS:
                gk = float(g[f"mse_{estimand}"])
                dk = float(d[f"mse_{estimand}"])
                cmp[f"gnn_mse_{estimand}"] = gk
                cmp[f"dirgnn_mse_{estimand}"] = dk
                cmp[f"delta_mse_{estimand}"] = gk - dk
                cmp[f"ratio_mse_{estimand}"] = (gk / dk) if dk > 0 else float("inf")
            compare_rows.append(cmp)
            print(
                f"theta={theta_name} mean_mse_all gnn={float(g['mean_mse_all']):.6f} "
                f"dirgnn={float(d['mean_mse_all']):.6f} delta={float(cmp['delta_mean_mse_all']):.6f}",
                flush=True,
            )
    finally:
        gen_data.THETA_D_DIR = original_theta

    if not compare_rows:
        raise RuntimeError(
            "No theta settings passed smoke check. "
            "Try increasing --max_treated_prop or adjusting THETA_CANDIDATES."
        )

    compare_rows.sort(key=lambda r: float(r["delta_mean_mse_all"]), reverse=True)
    for rank, row in enumerate(compare_rows, start=1):
        row["rank_dirgnn_advantage"] = int(rank)

    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_absolute():
        summary_csv = ROOT / summary_csv
    compare_csv = Path(args.compare_csv)
    if not compare_csv.is_absolute():
        compare_csv = ROOT / compare_csv

    write_csv(summary_rows, summary_csv)
    write_csv(compare_rows, compare_csv)
    return summary_csv, compare_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep THETA_D_DIR settings and compare GNN vs DirGNN MSE on spillover DGP (rgg)."
        )
    )
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=123)
    parser.add_argument("--p_bidirected", type=float, default=0.05)
    parser.add_argument("--tau_dir_true", type=float, default=-2.0)
    parser.add_argument("--tau_in_true", type=float, default=1.0)
    parser.add_argument("--tau_out_true", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=0.001)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--log_every_run", type=int, default=2)
    parser.add_argument("--smoke_trials", type=int, default=3)
    parser.add_argument("--smoke_seed_start", type=int, default=900000)
    parser.add_argument("--max_treated_prop", type=float, default=0.7)
    parser.add_argument("--smoke_fail_action", type=str, choices=("skip", "error"), default="skip")
    parser.add_argument("--summary_csv", type=str, default="paratune/theta_dir_model_mse_summary.csv")
    parser.add_argument("--compare_csv", type=str, default="paratune/theta_dir_model_mse_compare.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    start = time.time()
    summary_csv, compare_csv = run(args)
    elapsed = time.time() - start
    print(f"wrote summary: {summary_csv}", flush=True)
    print(f"wrote compare: {compare_csv}", flush=True)
    print(f"done in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
