#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ate import tau_vector_and_se_from_gnn
from src.gen_data import sample_data_spillover

ESTIMANDS = ("tau_dir", "tau_in", "tau_out", "tau_tot")


@dataclass(frozen=True)
class TuneConfig:
    config_id: str
    clip: float
    output_dim: int
    bandwidth_label: str
    bandwidth_value: int | None


def parse_float_list(raw: str) -> list[float]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return [float(item) for item in items]


def parse_int_list(raw: str) -> list[int]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated int list.")
    return [int(item) for item in items]


def parse_bandwidth_values(raw: str) -> list[tuple[str, int | None]]:
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated bandwidth list.")

    out: list[tuple[str, int | None]] = []
    for item in items:
        if item == "auto":
            out.append(("auto", None))
        else:
            value = int(item)
            if value <= 0:
                raise ValueError("Bandwidth must be positive when specified.")
            out.append((str(value), value))
    return out


def zscore(values: np.ndarray) -> np.ndarray:
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if sigma <= 1e-12:
        return np.zeros_like(values)
    return (values - mu) / sigma


def build_configs(
    clip_values: list[float],
    output_dims: list[int],
    bandwidth_values: list[tuple[str, int | None]],
) -> list[TuneConfig]:
    configs: list[TuneConfig] = []
    for idx, (clip, output_dim, bw) in enumerate(
        itertools.product(clip_values, output_dims, bandwidth_values),
        start=1,
    ):
        bw_label, bw_value = bw
        configs.append(
            TuneConfig(
                config_id=f"cfg_{idx:03d}",
                clip=float(clip),
                output_dim=int(output_dim),
                bandwidth_label=str(bw_label),
                bandwidth_value=bw_value,
            )
        )
    return configs


def write_csv(rows: list[dict], output_csv: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    clip_values = parse_float_list(args.clip_values)
    output_dims = parse_int_list(args.output_dims)
    bandwidth_values = parse_bandwidth_values(args.bandwidth_values)
    configs = build_configs(clip_values, output_dims, bandwidth_values)

    var_method = args.variance_method.strip() or None
    use_cuda = bool(int(args.use_gpu)) and torch.cuda.is_available()
    z_crit = float(NormalDist().inv_cdf(1.0 - float(args.ci_alpha) / 2.0))

    print(
        f"running {len(configs)} configs | graph_model=rgg | num_runs={args.num_runs} | n={args.n} "
        f"| variance_type={args.variance_type} | variance_method={var_method or 'default'} "
        f"| z_crit={z_crit:.5f} | device={'cuda' if use_cuda else 'cpu'}",
        flush=True,
    )

    rows: list[dict] = []
    for cfg_idx, cfg in enumerate(configs, start=1):
        cfg_start = time.time()
        sq_err: dict[str, list[float]] = {k: [] for k in ESTIMANDS}
        cover_hits: dict[str, list[int]] = {k: [] for k in ESTIMANDS}
        se_vals: dict[str, list[float]] = {k: [] for k in ESTIMANDS}

        for run_idx in range(int(args.num_runs)):
            draw_seed = int(args.seed_start) + run_idx
            draw = sample_data_spillover(
                sample_size=int(args.n),
                seed=draw_seed,
                graph_model="rgg",
                p_bidirected=float(args.p_bidirected),
                tau_dir=float(args.tau_dir_true),
                tau_in=float(args.tau_in_true),
                tau_out=float(args.tau_out_true),
            )

            fit = tau_vector_and_se_from_gnn(
                draw,
                feature_key="node_features",
                clip=float(cfg.clip),
                num_layers=int(args.num_layers),
                output_dim=int(cfg.output_dim),
                seed=draw_seed,
                directed=True,
                use_gpu=use_cuda,
                variance_type=str(args.variance_type),
                variance_method=var_method,
                bandwidth=cfg.bandwidth_value,
            )

            true_taus = draw["true_taus"]
            for estimand in ESTIMANDS:
                tau_hat = float(fit["tau_hat"][estimand])
                se_hat = float(fit["se_hat"][estimand])
                tau_true = float(true_taus[estimand])
                sq_err[estimand].append((tau_hat - tau_true) ** 2)
                se_vals[estimand].append(se_hat)
                hw = z_crit * se_hat
                cover_hits[estimand].append(int((tau_hat - hw) <= tau_true <= (tau_hat + hw)))

        row: dict[str, float | int | str] = {
            "config_id": cfg.config_id,
            "clip": float(cfg.clip),
            "output_dim": int(cfg.output_dim),
            "bandwidth": cfg.bandwidth_label,
            "bandwidth_is_auto": int(cfg.bandwidth_value is None),
            "num_layers": int(args.num_layers),
            "n": int(args.n),
            "num_runs": int(args.num_runs),
            "ci_alpha": float(args.ci_alpha),
            "z_crit": float(z_crit),
            "variance_type": str(args.variance_type),
            "variance_method": str(var_method or ""),
            "runtime_sec": float(time.time() - cfg_start),
        }

        mse_all = []
        ci_gap_all = []
        cover_all = []
        for estimand in ESTIMANDS:
            mse_e = float(np.mean(np.asarray(sq_err[estimand], dtype=float)))
            rmse_e = float(np.sqrt(mse_e))
            cover_e = float(np.mean(np.asarray(cover_hits[estimand], dtype=float)))
            mean_se_e = float(np.mean(np.asarray(se_vals[estimand], dtype=float)))
            ci_gap_e = abs(cover_e - (1.0 - float(args.ci_alpha)))

            row[f"mse_{estimand}"] = mse_e
            row[f"rmse_{estimand}"] = rmse_e
            row[f"mean_se_{estimand}"] = mean_se_e
            row[f"cover_rate_{estimand}"] = cover_e
            row[f"ci_gap_{estimand}"] = ci_gap_e

            mse_all.append(mse_e)
            ci_gap_all.append(ci_gap_e)
            cover_all.append(cover_e)

        row["mean_mse_all"] = float(np.mean(np.asarray(mse_all, dtype=float)))
        row["mean_ci_gap_all"] = float(np.mean(np.asarray(ci_gap_all, dtype=float)))
        row["mean_cover_all"] = float(np.mean(np.asarray(cover_all, dtype=float)))
        rows.append(row)

        print(
            f"[{cfg_idx:02d}/{len(configs)}] {cfg.config_id} "
            f"clip={cfg.clip} output_dim={cfg.output_dim} bw={cfg.bandwidth_label} "
            f"mean_mse_all={row['mean_mse_all']:.6f} mean_cover_all={row['mean_cover_all']:.3f} "
            f"mean_ci_gap_all={row['mean_ci_gap_all']:.3f} time={row['runtime_sec']:.1f}s",
            flush=True,
        )

    mse_vec = np.asarray([float(row["mean_mse_all"]) for row in rows], dtype=float)
    gap_vec = np.asarray([float(row["mean_ci_gap_all"]) for row in rows], dtype=float)
    z_mse = zscore(mse_vec)
    z_gap = zscore(gap_vec)

    for i, row in enumerate(rows):
        row["z_mean_mse_all"] = float(z_mse[i])
        row["z_mean_ci_gap_all"] = float(z_gap[i])
        row["joint_score"] = 0.5 * float(z_mse[i]) + 0.5 * float(z_gap[i])

    rows.sort(key=lambda item: float(item["joint_score"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    write_csv(rows, output_csv)

    top_k = min(int(args.print_top), len(rows))
    print("top configs:", flush=True)
    for row in rows[:top_k]:
        print(
            f"rank={row['rank']:02d} {row['config_id']} clip={row['clip']} od={row['output_dim']} "
            f"bw={row['bandwidth']} score={row['joint_score']:.4f} "
            f"mse_all={row['mean_mse_all']:.6f} cover_all={row['mean_cover_all']:.3f} "
            f"gap_all={row['mean_ci_gap_all']:.3f}",
            flush=True,
        )

    return output_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune DirGNN on RGG by comparing clipping, output_dim, and bandwidth using "
            "Monte Carlo MSE and CI coverage."
        )
    )
    parser.add_argument("--num_runs", type=int, default=50)
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed_start", type=int, default=123)

    parser.add_argument("--tau_dir_true", type=float, default=-2.0)
    parser.add_argument("--tau_in_true", type=float, default=1.0)
    parser.add_argument("--tau_out_true", type=float, default=1.0)
    parser.add_argument("--p_bidirected", type=float, default=0.05)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--clip_values", type=str, default="1e-4,5e-4,1e-3,5e-3,1e-2")
    parser.add_argument("--output_dims", type=str, default="4,6,8")
    parser.add_argument("--bandwidth_values", type=str, default="auto,2,3,4")

    parser.add_argument("--variance_type", type=str, default="directed", choices=("iid", "skeleton", "directed"))
    parser.add_argument("--variance_method", type=str, default="dir_avg")
    parser.add_argument("--ci_alpha", type=float, default=0.05)
    parser.add_argument("--use_gpu", type=int, default=0)

    parser.add_argument("--output_csv", type=str, default="paratune/dirgnn_rgg_mse_ci_tune.csv")
    parser.add_argument("--print_top", type=int, default=10)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    start = time.time()
    output_csv = run(args)
    elapsed = time.time() - start
    print(f"wrote tuning table: {output_csv} ({elapsed:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
