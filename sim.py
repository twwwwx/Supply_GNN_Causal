#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import csv
import time
import warnings

import numpy as np
from tqdm import tqdm

from src.baseline import estimate_tau_hat_dr_linear
from src.gen_data import sample_data_undir, sample_data_simple, sample_data_dir

# Keep repeated PyG import warnings compact in logs.
warnings.filterwarnings("once", category=UserWarning, module=r"torch_geometric")


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = [
        "model",
        "DGP",
        "num_runs",
        "n",
        "seed",
        "gen_graph",
        "tau_true",
        "p_treat",
        "features",
        "clip",
        "L",
        "output_dim",
        "variance_type",
        "variance_method",
        "bandwidth",
        "MSE",
        "se_tau_hat_MC",
        "mean_sigma_hat",
        "cover_rate",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def draw_data(DGP: str, n: int, seed: int, gen_graph: str, tau_true: float, p_treat: float):
    if DGP == "undir":
        return sample_data_undir(sample_size=n, seed=seed, graph_model=gen_graph)
    if DGP == "simple_undir":
        return sample_data_simple(
            sample_size=n,
            seed=seed,
            graph_model=gen_graph,
            tau=tau_true,
            p_treat=p_treat,
        )
    if DGP == "dir":
        return sample_data_dir(
            sample_size=n,
            seed=seed,
            graph_model=gen_graph,
        )
    raise ValueError("DGP must be one of: undir, simple_undir, dir.")


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
    if model == "linear":
        fit = estimate_tau_hat_dr_linear(draw, feature_key=features, clip=clip)
        return {
            "tau_hat": float(fit["tau_hat"]),
            "se_hat": None,
            "variance_type": "none",
            "variance_method": "none",
            "bandwidth": None,
        }
    if model in {"gnn", "dirgnn"}:
        from src.ate import tau_hat_and_se_from_gnn

        fit = tau_hat_and_se_from_gnn(
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
            "tau_hat": float(fit["tau_hat"]),
            "se_hat": float(fit["se_hat"]),
            "variance_type": str(fit["variance_type"]),
            "variance_method": str(fit["variance_method"]),
            "bandwidth": fit["bandwidth"],
        }
    raise ValueError("model must be one of: linear, gnn, dirgnn.")


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="linear", help="linear, gnn, or dirgnn")
    parser.add_argument("--DGP", type=str, default="simple_undir", help="undir, simple_undir, or dir")
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gen_graph", type=str, default="rgg")
    parser.add_argument("--tau_true", type=float, default=2.0)
    parser.add_argument("--p_treat", type=float, default=0.5)
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
    DGP = args.DGP.lower()

    if model not in {"linear", "gnn", "dirgnn"}:
        raise ValueError("--model must be linear, gnn, or dirgnn.")
    if DGP not in {"undir", "simple_undir", "dir"}:
        raise ValueError("--DGP must be undir, simple_undir, or dir.")
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

    cuda_available = False
    actual_device = "cpu"
    if model in {"gnn", "dirgnn"}:
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

    if model in {"gnn", "dirgnn"} and features_arg == "X":
        features = "node_features"
    else:
        features = features_arg

    root_dir = Path(__file__).resolve().parent
    metrics_csv = Path(args.metrics_csv)
    if not metrics_csv.is_absolute():
        metrics_csv = root_dir / metrics_csv

    tau_hats = []
    se_hats = []
    cover_hits = []
    bw_hats = []
    start_time = time.time()
    run_iter = range(args.num_runs)
    if model in {"gnn", "dirgnn"}:
        run_iter = tqdm(run_iter, total=args.num_runs, desc=f"{model}-eval")

    for i in run_iter:
        draw_seed = args.seed + i
        draw = draw_data(
            DGP=DGP,
            n=args.n,
            seed=draw_seed,
            gen_graph=args.gen_graph,
            tau_true=args.tau_true,
            p_treat=args.p_treat,
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
        tau_hats.append(float(fit["tau_hat"]))
        if fit["se_hat"] is not None:
            se_hat_i = float(fit["se_hat"])
            se_hats.append(se_hat_i)
            tau_hat_i = float(fit["tau_hat"])
            ci_half_width = 1.96 * se_hat_i
            cover_hits.append(int((tau_hat_i - ci_half_width) <= float(args.tau_true) <= (tau_hat_i + ci_half_width)))
        if fit["bandwidth"] is not None:
            bw_hats.append(float(fit["bandwidth"]))

    tau_hats_arr = np.asarray(tau_hats, dtype=float)
    sq_err = (tau_hats_arr - float(args.tau_true)) ** 2
    mse_tau_hat = round(float(np.mean(sq_err)), 3)
    if int(args.num_runs) > 1:
        se_mse_tau_hat = float(np.std(sq_err, ddof=1) / np.sqrt(float(args.num_runs)))
    else:
        se_mse_tau_hat = 0.0
    mean_tau_hat = round(float(np.mean(tau_hats_arr)), 3)
    if int(args.num_runs) > 1:
        se_tau_hat = float(np.std(tau_hats_arr, ddof=1) / np.sqrt(float(args.num_runs)))
    else:
        se_tau_hat = 0.0
    mean_se_hat = float(np.mean(np.asarray(se_hats, dtype=float))) if se_hats else 0.0
    cover_rate = float(np.mean(np.asarray(cover_hits, dtype=float))) if cover_hits else 0.0
    mean_bw_hat = float(np.mean(np.asarray(bw_hats, dtype=float))) if bw_hats else 0.0
    variance_type_out = variance_type if model in {"gnn", "dirgnn"} else "none"
    variance_method_out = (variance_method or "") if model in {"gnn", "dirgnn"} else ""
    bandwidth_out = args.bandwidth if args.bandwidth is not None else (round(mean_bw_hat, 3) if bw_hats else "")

    append_row(
        metrics_csv,
        {
            "model": model,
            "DGP": DGP,
            "num_runs": int(args.num_runs),
            "n": int(args.n),
            "seed": int(args.seed),
            "gen_graph": args.gen_graph,
            "tau_true": float(args.tau_true),
            "p_treat": float(args.p_treat),
            "features": features,
            "clip": float(args.clip),
            "L": int(args.L),
            "output_dim": int(args.output_dim),
            "variance_type": variance_type_out,
            "variance_method": variance_method_out,
            "bandwidth": bandwidth_out,
            "MSE": f"{mse_tau_hat:.3f} ({se_mse_tau_hat:.6f})",
            "se_tau_hat_MC": round(se_tau_hat, 6),
            "mean_sigma_hat": round(mean_se_hat, 6),
            "cover_rate": round(cover_rate, 6),
        },
    )
    print(
        f"saved n={args.n} model={model} DGP={DGP} "
        f"mean_tau_hat={mean_tau_hat:.3f} mse_tau_hat={mse_tau_hat:.3f} "
        f"se_mse_tau_hat={se_mse_tau_hat:.6f} "
        f"se_tau_hat={se_tau_hat:.6f} mean_se_hat={mean_se_hat:.6f} "
        f"cover_rate={cover_rate:.6f} "
        f"var_type={variance_type_out} var_method={variance_method_out} bw={bandwidth_out}"
    )
    elapsed = time.time() - start_time
    print(f"evaluation_time_sec={elapsed:.3f} evaluation_time_min={elapsed/60.0:.3f}")
    print(f"Wrote 1 row to {metrics_csv}")


if __name__ == "__main__":
    main()
