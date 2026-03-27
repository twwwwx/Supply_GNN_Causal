#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import csv
import time
import warnings

import numpy as np
from tqdm import tqdm

from dGC.baseline import estimate_tau_hat_dr_linear
from dGC.gen_data import sample_data_undir, sample_data_simple, sample_data_dir

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
        "mean_tau_hat",
        "mse_tau_hat",
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


def tau_hat_from_model(model: str, draw: dict, features: str, clip: float, L: int, output_dim: int, seed: int) -> float:
    if model == "linear":
        fit = estimate_tau_hat_dr_linear(draw, feature_key=features, clip=clip)
        return float(fit["tau_hat"])
    if model in {"gnn", "dirgnn"}:
        from dGC.ate import tau_hat_from_gnn

        return float(
            tau_hat_from_gnn(
                draw,
                feature_key=features,
                clip=clip,
                num_layers=L,
                output_dim=output_dim,
                seed=seed,
                directed=(model == "dirgnn"),
            )
        )
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
    parser.add_argument("--metrics_csv", type=str, default="results/metrics.csv")
    args = parser.parse_args()

    model = args.model.lower()
    DGP = args.DGP.lower()

    if model not in {"linear", "gnn", "dirgnn"}:
        raise ValueError("--model must be linear, gnn, or dirgnn.")
    if DGP not in {"undir", "simple_undir", "dir"}:
        raise ValueError("--DGP must be undir, simple_undir, or dir.")

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
        tau_hat = tau_hat_from_model(
            model=model,
            draw=draw,
            features=features,
            clip=args.clip,
            L=args.L,
            output_dim=args.output_dim,
            seed=draw_seed,
        )
        tau_hats.append(tau_hat)

    tau_hats_arr = np.asarray(tau_hats, dtype=float)
    mse_tau_hat = round(float(np.mean((tau_hats_arr - float(args.tau_true)) ** 2)), 3)
    mean_tau_hat = round(float(np.mean(tau_hats_arr)), 3)

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
            "mean_tau_hat": mean_tau_hat,
            "mse_tau_hat": mse_tau_hat,
        },
    )
    print(
        f"saved n={args.n} model={model} DGP={DGP} "
        f"mean_tau_hat={mean_tau_hat:.3f} mse_tau_hat={mse_tau_hat:.3f}"
    )
    elapsed = time.time() - start_time
    print(f"evaluation_time_sec={elapsed:.3f} evaluation_time_min={elapsed/60.0:.3f}")
    print(f"Wrote 1 row to {metrics_csv}")


if __name__ == "__main__":
    main()
