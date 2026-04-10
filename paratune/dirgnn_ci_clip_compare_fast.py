#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ate import (  # noqa: E402
    BASELINE_STATE,
    ESTIMAND_TARGETS,
    STATE_TO_INDEX,
    _clip_and_renormalize_multiclass_probs,
    _contrast_scores,
)
from src.GNN import GNN_reg_dir_multiclass, GNN_reg_dir_outcome_surface  # noqa: E402
from src.gen_data import sample_data_spillover  # noqa: E402
from src.variance import _all_pairs_shortest_paths, _pd_kernel_from_mask, select_bandwidth  # noqa: E402

ESTIMANDS = ("tau_dir", "tau_in", "tau_out", "tau_tot")
EXPOSURE_STATES = list(STATE_TO_INDEX.keys())


def parse_float_list(raw: str) -> list[float]:
    vals = [item.strip() for item in raw.split(",") if item.strip()]
    if not vals:
        raise ValueError("clip list cannot be empty.")
    return [float(v) for v in vals]


def _u_kernel_from_dist(dist: np.ndarray, bandwidth: int) -> np.ndarray:
    b = int(bandwidth)
    return (np.isfinite(dist) & (dist <= b)).astype(float)


def _variance_from_kernel(tau_tilde: np.ndarray, kernel: np.ndarray, m_n: float) -> float:
    t = np.asarray(tau_tilde, dtype=float).reshape(-1)
    return float((t @ kernel @ t) / m_n)


def build_directed_kernels(adjacency: np.ndarray, bandwidth: int | None) -> tuple[int, dict[str, np.ndarray]]:
    a = (np.asarray(adjacency) != 0).astype(np.int8)
    b = select_bandwidth(a, directed=True) if bandwidth is None else int(bandwidth)
    dist_out = _all_pairs_shortest_paths(a)

    k_u_out = _u_kernel_from_dist(dist_out, b)
    k_u_in = _u_kernel_from_dist(dist_out.T, b)

    half_b = max(1, int(math.floor(b / 2)))
    out_mask = dist_out <= half_b
    in_mask = dist_out.T <= half_b
    k_pd_out = _pd_kernel_from_mask(out_mask)
    k_pd_in = _pd_kernel_from_mask(in_mask)
    return b, {
        "k_u_in": k_u_in,
        "k_u_out": k_u_out,
        "k_pd_in": k_pd_in,
        "k_pd_out": k_pd_out,
    }


def sigma2_directed(
    tau_tilde: np.ndarray,
    kernels: dict[str, np.ndarray],
    m_n: float,
    method: str,
) -> float:
    vmethod = method.lower()
    if vmethod not in {"in_max", "out_max", "dir_max", "dir_avg"}:
        raise ValueError("variance_method must be in {in_max,out_max,dir_max,dir_avg}.")

    sigma2_u_in = _variance_from_kernel(tau_tilde, kernels["k_u_in"], m_n)
    sigma2_u_out = _variance_from_kernel(tau_tilde, kernels["k_u_out"], m_n)
    sigma2_pd_in = _variance_from_kernel(tau_tilde, kernels["k_pd_in"], m_n)
    sigma2_pd_out = _variance_from_kernel(tau_tilde, kernels["k_pd_out"], m_n)

    sigma2_in_max = max(sigma2_u_in, sigma2_pd_in)
    sigma2_out_max = max(sigma2_u_out, sigma2_pd_out)
    sigma2_dir_max = max(sigma2_out_max, sigma2_in_max)
    sigma2_dir_avg = 0.5 * (sigma2_out_max + sigma2_in_max)
    sigma2_map = {
        "in_max": sigma2_in_max,
        "out_max": sigma2_out_max,
        "dir_max": sigma2_dir_max,
        "dir_avg": sigma2_dir_avg,
    }
    return float(sigma2_map[vmethod])


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    clips = parse_float_list(args.clip_values)
    use_gpu = bool(int(args.use_gpu))
    z_crit = float(NormalDist().inv_cdf(1.0 - float(args.ci_alpha) / 2.0))

    print(
        f"clip_compare_fast runs={args.num_runs} n={args.n} clips={clips} "
        f"output_dim={args.output_dim} bw={args.bandwidth} method={args.variance_method} "
        f"dropout=0(by src DirGNN) use_gpu={int(use_gpu)}",
        flush=True,
    )

    acc: dict[float, dict[str, list[float]]] = {}
    for clip in clips:
        acc[clip] = {}
        for key in ESTIMANDS:
            acc[clip][f"mse_{key}"] = []
            acc[clip][f"cover_{key}"] = []
            acc[clip][f"se_{key}"] = []
        acc[clip]["bw"] = []

    base_idx = int(STATE_TO_INDEX[BASELINE_STATE])
    states_order = list(STATE_TO_INDEX.keys())
    bandwidth_arg = None if str(args.bandwidth).lower() == "auto" else int(args.bandwidth)

    total_runs = int(args.num_runs)
    started = time.time()
    for run_idx in range(total_runs):
        seed = int(args.seed_start) + run_idx
        draw = sample_data_spillover(
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
        a = np.asarray(draw["adjacency"])
        t_obs = np.asarray(draw["T"], dtype=float)
        state_obs = np.asarray(draw["state_index"], dtype=int).reshape(-1)
        true_taus = draw["true_taus"]
        n_float = float(y.size)

        # Nuisance fits once per draw (shared across clips).
        e_hat_raw = np.asarray(
            GNN_reg_dir_multiclass(
                labels=state_obs,
                X=x,
                A=a,
                num_classes=8,
                num_layers=int(args.num_layers),
                output_dim=int(args.output_dim),
                seed=seed,
                use_gpu=use_gpu,
            ),
            dtype=float,
        )
        mu_hat = np.asarray(
            GNN_reg_dir_outcome_surface(
                Y=y,
                X=x,
                A=a,
                exposure_obs=t_obs,
                states=states_order,
                num_layers=int(args.num_layers),
                output_dim=int(args.output_dim),
                seed=seed + 1,
                use_gpu=use_gpu,
            ),
            dtype=float,
        )

        bw_used, kernels = build_directed_kernels(a, bandwidth_arg)
        for clip in clips:
            e_hat = _clip_and_renormalize_multiclass_probs(e_hat_raw, clip=clip)
            acc[clip]["bw"].append(float(bw_used))

            for estimand, target_state in ESTIMAND_TARGETS.items():
                target_idx = int(STATE_TO_INDEX[target_state])
                contrast = _contrast_scores(
                    y=y,
                    observed_state_idx=state_obs,
                    mu_hat=mu_hat,
                    e_hat=e_hat,
                    target_idx=target_idx,
                    baseline_idx=base_idx,
                )
                tau_hat = float(contrast["tau_hat"])
                tau_true = float(true_taus[estimand])
                tau_tilde = np.asarray(contrast["tau_tilde"], dtype=float)

                sigma2 = sigma2_directed(
                    tau_tilde=tau_tilde,
                    kernels=kernels,
                    m_n=n_float,
                    method=str(args.variance_method),
                )
                se = float(math.sqrt(max(sigma2, 0.0) / n_float))
                hw = z_crit * se
                covered = int((tau_hat - hw) <= tau_true <= (tau_hat + hw))

                acc[clip][f"mse_{estimand}"].append((tau_hat - tau_true) ** 2)
                acc[clip][f"cover_{estimand}"].append(covered)
                acc[clip][f"se_{estimand}"].append(se)

        if (run_idx + 1) % int(args.log_every) == 0 or (run_idx + 1) == total_runs:
            elapsed = time.time() - started
            print(f"completed_runs={run_idx + 1}/{total_runs} elapsed_sec={elapsed:.1f}", flush=True)

    rows: list[dict] = []
    for clip in clips:
        row: dict[str, float | int | str] = {
            "clip": float(clip),
            "n": int(args.n),
            "num_runs": int(args.num_runs),
            "output_dim": int(args.output_dim),
            "num_layers": int(args.num_layers),
            "bandwidth": str(args.bandwidth),
            "mean_bandwidth_used": float(np.mean(np.asarray(acc[clip]["bw"], dtype=float))),
            "variance_method": str(args.variance_method),
            "ci_alpha": float(args.ci_alpha),
            "z_crit": float(z_crit),
        }

        mse_all = []
        cover_all = []
        gap_all = []
        for estimand in ESTIMANDS:
            mse_e = float(np.mean(np.asarray(acc[clip][f"mse_{estimand}"], dtype=float)))
            cover_e = float(np.mean(np.asarray(acc[clip][f"cover_{estimand}"], dtype=float)))
            mean_se_e = float(np.mean(np.asarray(acc[clip][f"se_{estimand}"], dtype=float)))
            gap_e = abs(cover_e - (1.0 - float(args.ci_alpha)))

            row[f"mse_{estimand}"] = mse_e
            row[f"cover_rate_{estimand}"] = cover_e
            row[f"mean_se_{estimand}"] = mean_se_e
            row[f"ci_gap_{estimand}"] = gap_e

            mse_all.append(mse_e)
            cover_all.append(cover_e)
            gap_all.append(gap_e)

        row["mean_mse_all"] = float(np.mean(np.asarray(mse_all, dtype=float)))
        row["mean_cover_all"] = float(np.mean(np.asarray(cover_all, dtype=float)))
        row["mean_ci_gap_all"] = float(np.mean(np.asarray(gap_all, dtype=float)))
        rows.append(row)

    rows.sort(key=lambda r: (float(r["mean_ci_gap_all"]), float(r["mean_mse_all"])))
    for rank, row in enumerate(rows, start=1):
        row["rank_ci_then_mse"] = rank

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    write_csv(rows, output_csv)

    print("top by CI-gap then MSE:", flush=True)
    for row in rows[: min(3, len(rows))]:
        print(
            f"rank={row['rank_ci_then_mse']} clip={row['clip']} "
            f"cover_all={row['mean_cover_all']:.3f} gap_all={row['mean_ci_gap_all']:.3f} "
            f"tau_tot_cover={row['cover_rate_tau_tot']:.3f} mse_all={row['mean_mse_all']:.6f}",
            flush=True,
        )
    return output_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fast CI comparison over clip values for DirGNN on RGG at fixed output_dim and bandwidth. "
            "Caches nuisance fits and directed kernels per run."
        )
    )
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed_start", type=int, default=123)

    parser.add_argument("--tau_dir_true", type=float, default=-2.0)
    parser.add_argument("--tau_in_true", type=float, default=1.0)
    parser.add_argument("--tau_out_true", type=float, default=1.0)
    parser.add_argument("--p_bidirected", type=float, default=0.05)

    parser.add_argument("--clip_values", type=str, default="0.01,0.005,0.001")
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--bandwidth", type=str, default="auto")
    parser.add_argument("--variance_method", type=str, default="dir_avg")
    parser.add_argument("--ci_alpha", type=float, default=0.05)
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--output_csv", type=str, default="paratune/dirgnn_ci_clip_compare_fast.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    start = time.time()
    output_csv = run(args)
    elapsed = time.time() - start
    print(f"wrote: {output_csv} ({elapsed:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
