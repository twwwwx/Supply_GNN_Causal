#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import csv
import time
import warnings

import numpy as np
from tqdm import tqdm

from src.gen_data import sample_data_spillover

# Keep repeated PyG import warnings compact in logs.
warnings.filterwarnings("once", category=UserWarning, module=r"torch_geometric")

ESTIMANDS = ("tau_dir", "tau_in", "tau_out", "tau_tot")


def _csv_fieldnames() -> list[str]:
    base = [
        "model",
        "DGP",
        "num_runs",
        "n",
        "seed",
        "gen_graph",
        "tau_dir_true",
        "tau_in_true",
        "tau_out_true",
        "tau_tot_true",
        "features",
        "clip",
        "L",
        "output_dim",
        "variance_type",
        "variance_method",
        "bandwidth",
    ]
    compact_tau = list(ESTIMANDS)
    mean_sigma = [f"mean_sigma_hat_{name}" for name in ESTIMANDS]
    mse_cols = [f"MSE_{name}" for name in ESTIMANDS]
    cover_cols = [f"cover_rate_{name}" for name in ESTIMANDS]
    return base + compact_tau + mean_sigma + mse_cols + cover_cols


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    needs_header = (not file_exists) or (csv_path.stat().st_size == 0)
    fieldnames = _csv_fieldnames()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def draw_data(
    n: int,
    seed: int,
    gen_graph: str,
    tau_dir_true: float,
    tau_in_true: float,
    tau_out_true: float,
) -> dict:
    return sample_data_spillover(
        sample_size=n,
        seed=seed,
        graph_model=gen_graph,
        tau_dir=tau_dir_true,
        tau_in=tau_in_true,
        tau_out=tau_out_true,
    )


def estimate_from_model(
    model: str,
    draw: dict,
    features: str,
    clip: float,
    L: int,
    output_dim: int,
    seed: int,
    variance_type: str,
    variance_method: str | None,
    bandwidth: int | None,
    use_gpu: bool,
) -> dict:
    if model not in {"gnn", "dirgnn"}:
        raise ValueError("model must be one of: gnn, dirgnn.")
    from src.ate import tau_vector_and_se_from_gnn

    fit = tau_vector_and_se_from_gnn(
        draw,
        feature_key=features,
        clip=clip,
        num_layers=L,
        output_dim=output_dim,
        seed=seed,
        directed=(model == "dirgnn"),
        use_gpu=use_gpu,
        variance_type=variance_type,
        variance_method=variance_method,
        bandwidth=bandwidth,
    )
    return {
        "tau_hat": {k: float(v) for k, v in fit["tau_hat"].items()},
        "se_hat": {k: float(v) for k, v in fit["se_hat"].items()},
        "variance_type": str(fit["variance_type"]),
        "variance_method": {k: str(v) for k, v in fit["variance_method"].items()},
        "bandwidth": dict(fit["bandwidth"]),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="dirgnn", help="gnn or dirgnn")
    parser.add_argument("--DGP", type=str, default="spillover", help="spillover")
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gen_graph", type=str, default="rgg")
    parser.add_argument("--tau_dir_true", type=float, default=2.0)
    parser.add_argument("--tau_in_true", type=float, default=1.0)
    parser.add_argument("--tau_out_true", type=float, default=1.0)
    parser.add_argument("--features", type=str, default="nodes")
    parser.add_argument("--clip", type=float, default=1e-3)
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--variance_type", type=str, default="skeleton", help="iid, skeleton, or directed")
    parser.add_argument("--variance_method", type=str, default="", help="Kernel method inside variance type")
    parser.add_argument("--bandwidth", type=int, default=None, help="Optional fixed bandwidth")
    parser.add_argument("--use_gpu", type=int, default=1, help="1 to allow CUDA for GNN models, 0 to force CPU")
    parser.add_argument("--metrics_csv", type=str, default="results/metrics.csv")
    args = parser.parse_args()

    model = args.model.lower()
    dgp = args.DGP.lower()
    if model not in {"gnn", "dirgnn"}:
        raise ValueError("--model must be gnn or dirgnn.")
    if dgp != "spillover":
        raise ValueError("--DGP must be spillover.")

    variance_type = args.variance_type.lower()
    if variance_type not in {"iid", "skeleton", "directed"}:
        raise ValueError("--variance_type must be iid, skeleton, or directed.")
    variance_method = args.variance_method.strip() or None
    use_gpu = bool(int(args.use_gpu))

    if model == "gnn" and variance_type == "directed":
        warnings.warn(
            "model='gnn' is undirected; variance_type='directed' is overridden to 'skeleton'.",
            UserWarning,
        )
        variance_type = "skeleton"
        if variance_method is None or variance_method.startswith("dir"):
            variance_method = "max"

    import torch

    cuda_available = bool(torch.cuda.is_available())
    actual_device = "cuda" if (use_gpu and cuda_available) else "cpu"
    print(
        f"device_config model={model} use_gpu_arg={int(use_gpu)} "
        f"cuda_available={int(cuda_available)} using_device={actual_device}"
    )

    features_arg = args.features
    if features_arg == "nodes":
        features_arg = "node_features"
    if features_arg == "X":
        features = "node_features"
    else:
        features = features_arg

    root_dir = Path(__file__).resolve().parent
    metrics_csv = Path(args.metrics_csv)
    if not metrics_csv.is_absolute():
        metrics_csv = root_dir / metrics_csv

    tau_true_map = {
        "tau_dir": float(args.tau_dir_true),
        "tau_in": float(args.tau_in_true),
        "tau_out": float(args.tau_out_true),
        "tau_tot": float(args.tau_dir_true + args.tau_in_true + args.tau_out_true),
    }

    tau_hats = {k: [] for k in ESTIMANDS}
    se_hats = {k: [] for k in ESTIMANDS}
    cover_hits = {k: [] for k in ESTIMANDS}
    bw_hats = {k: [] for k in ESTIMANDS}
    variance_method_map = {k: "" for k in ESTIMANDS}

    start_time = time.time()
    run_iter = tqdm(range(args.num_runs), total=args.num_runs, desc=f"{model}-eval")
    for i in run_iter:
        draw_seed = args.seed + i
        draw = draw_data(
            n=args.n,
            seed=draw_seed,
            gen_graph=args.gen_graph,
            tau_dir_true=args.tau_dir_true,
            tau_in_true=args.tau_in_true,
            tau_out_true=args.tau_out_true,
        )
        fit = estimate_from_model(
            model=model,
            draw=draw,
            features=features,
            clip=args.clip,
            L=args.L,
            output_dim=args.output_dim,
            seed=draw_seed,
            variance_type=variance_type,
            variance_method=variance_method,
            bandwidth=args.bandwidth,
            use_gpu=use_gpu,
        )

        for estimand in ESTIMANDS:
            tau_hat_i = float(fit["tau_hat"][estimand])
            se_hat_i = float(fit["se_hat"][estimand])
            tau_hats[estimand].append(tau_hat_i)
            se_hats[estimand].append(se_hat_i)
            ci_half_width = 1.96 * se_hat_i
            tau_true_i = tau_true_map[estimand]
            cover_hits[estimand].append(
                int((tau_hat_i - ci_half_width) <= tau_true_i <= (tau_hat_i + ci_half_width))
            )
            bw_i = fit["bandwidth"][estimand]
            if bw_i is not None:
                bw_hats[estimand].append(float(bw_i))
            variance_method_map[estimand] = str(fit["variance_method"][estimand])

    metrics_by_estimand: dict[str, dict] = {}
    for estimand in ESTIMANDS:
        tau_true_i = tau_true_map[estimand]
        tau_arr = np.asarray(tau_hats[estimand], dtype=float)
        se_arr = np.asarray(se_hats[estimand], dtype=float)
        cover_arr = np.asarray(cover_hits[estimand], dtype=float)
        bw_arr = np.asarray(bw_hats[estimand], dtype=float)

        sq_err = (tau_arr - tau_true_i) ** 2
        mse_tau_hat = round(float(np.mean(sq_err)), 3)
        mean_tau_hat = round(float(np.mean(tau_arr)), 3)
        if int(args.num_runs) > 1:
            se_mse_tau_hat = float(np.std(sq_err, ddof=1) / np.sqrt(float(args.num_runs)))
            se_tau_hat = float(np.std(tau_arr, ddof=1) / np.sqrt(float(args.num_runs)))
        else:
            se_mse_tau_hat = 0.0
            se_tau_hat = 0.0
        mean_se_hat = float(np.mean(se_arr)) if se_arr.size > 0 else 0.0
        cover_rate = float(np.mean(cover_arr)) if cover_arr.size > 0 else 0.0
        mean_bw_hat = float(np.mean(bw_arr)) if bw_arr.size > 0 else 0.0
        bandwidth_out = (
            args.bandwidth if args.bandwidth is not None else (round(mean_bw_hat, 3) if bw_arr.size > 0 else "")
        )
        metrics_by_estimand[estimand] = {
            "tau_compact": f"{mean_tau_hat:.3f}({se_tau_hat:.6f})",
            "mean_sigma_hat": round(mean_se_hat, 6),
            "mse": f"{mse_tau_hat:.3f}({se_mse_tau_hat:.6f})",
            "cover_rate": round(cover_rate, 6),
            "variance_method": variance_method_map[estimand],
            "bandwidth": bandwidth_out,
        }
        print(
            f"saved model={model} DGP={dgp} estimand={estimand} n={args.n} "
            f"mean_tau_hat={mean_tau_hat:.3f} mse_tau_hat={mse_tau_hat:.3f} "
            f"se_mse_tau_hat={se_mse_tau_hat:.6f} "
            f"se_tau_hat={se_tau_hat:.6f} mean_se_hat={mean_se_hat:.6f} "
            f"cover_rate={cover_rate:.6f} "
            f"var_type={variance_type} var_method={variance_method_map[estimand]} "
            f"bw={metrics_by_estimand[estimand]['bandwidth']}"
        )

    bw_candidates = [metrics_by_estimand[k]["bandwidth"] for k in ESTIMANDS if metrics_by_estimand[k]["bandwidth"] != ""]
    if not bw_candidates:
        bandwidth_out = ""
    else:
        unique_bw = sorted({str(v) for v in bw_candidates})
        bandwidth_out = unique_bw[0] if len(unique_bw) == 1 else "|".join(unique_bw)

    var_method_candidates = [
        metrics_by_estimand[k].get("variance_method", "")
        for k in ESTIMANDS
        if metrics_by_estimand[k].get("variance_method", "") != ""
    ]
    if not var_method_candidates:
        variance_method_out = ""
    else:
        unique_vm = sorted({str(v) for v in var_method_candidates})
        variance_method_out = unique_vm[0] if len(unique_vm) == 1 else "|".join(unique_vm)

    row = {
        "model": model,
        "DGP": dgp,
        "num_runs": int(args.num_runs),
        "n": int(args.n),
        "seed": int(args.seed),
        "gen_graph": args.gen_graph,
        "tau_dir_true": float(args.tau_dir_true),
        "tau_in_true": float(args.tau_in_true),
        "tau_out_true": float(args.tau_out_true),
        "tau_tot_true": float(args.tau_dir_true + args.tau_in_true + args.tau_out_true),
        "features": features,
        "clip": float(args.clip),
        "L": int(args.L),
        "output_dim": int(args.output_dim),
        "variance_type": variance_type,
        "variance_method": variance_method_out,
        "bandwidth": bandwidth_out,
    }
    for estimand in ESTIMANDS:
        row[estimand] = metrics_by_estimand[estimand]["tau_compact"]
        row[f"mean_sigma_hat_{estimand}"] = metrics_by_estimand[estimand]["mean_sigma_hat"]
        row[f"MSE_{estimand}"] = metrics_by_estimand[estimand]["mse"]
        row[f"cover_rate_{estimand}"] = metrics_by_estimand[estimand]["cover_rate"]

    append_row(metrics_csv, row)
    elapsed = time.time() - start_time
    print(f"evaluation_time_sec={elapsed:.3f} evaluation_time_min={elapsed/60.0:.3f}")
    print(f"Wrote 1 row to {metrics_csv}")


if __name__ == "__main__":
    main()
