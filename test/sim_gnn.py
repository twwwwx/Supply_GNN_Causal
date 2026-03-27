#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dGC.ate import tau_hat_from_gnn
from dGC.gen_data import sample_data_dir, sample_data_undir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Monte Carlo MSE of tau_hat (GNN DR estimator).")
    parser.add_argument("--num-preds", type=int, default=100, help="Number of Monte Carlo predictions.")
    parser.add_argument("--sample-size", type=int, default=100, help="Sample size per draw.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--graph-model", type=str, default="rgg", choices=["rgg", "er"], help="Graph model.")
    parser.add_argument("--feature-key", type=str, default="node_features", help="Feature key for the estimator.")
    parser.add_argument("--clip", type=float, default=1e-3, help="Propensity clipping threshold.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GNN layers.")
    parser.add_argument("--output-dim", type=int, default=6, help="Hidden width for each GNN layer.")
    parser.add_argument("--true-tau", type=float, default=None, help="Reference true tau; defaults to DGP value.")
    parser.add_argument("--directed", action="store_true", help="Use directed DGP + DirGNN.")
    parser.add_argument(
        "--p-bidirected",
        type=float,
        default=0.05,
        help="For directed DGP only: probability an undirected edge becomes bidirected.",
    )
    args = parser.parse_args()

    tau_hats = []
    tau_refs = []
    for i in range(args.num_preds):
        draw_seed = args.seed + i
        if args.directed:
            draw = sample_data_dir(
                sample_size=args.sample_size,
                seed=draw_seed,
                graph_model=args.graph_model,
                p_bidirected=args.p_bidirected,
            )
        else:
            draw = sample_data_undir(
                sample_size=args.sample_size,
                seed=draw_seed,
                graph_model=args.graph_model,
            )

        fit_tau = tau_hat_from_gnn(
            draw,
            feature_key=args.feature_key,
            clip=args.clip,
            num_layers=args.num_layers,
            output_dim=args.output_dim,
            seed=draw_seed + 10_000,
            directed=args.directed,
        )
        tau_hats.append(float(fit_tau))
        tau_refs.append(float(args.true_tau) if args.true_tau is not None else float(draw["true_tau"]))

    tau_arr = np.asarray(tau_hats, dtype=float)
    tau_ref_arr = np.asarray(tau_refs, dtype=float)
    mse_tau = float(np.mean((tau_arr - tau_ref_arr) ** 2))
    mode = "DirGNN" if args.directed else "GNN"
    print(f"MSE(tau_hat) over {args.num_preds} predictions [{mode}]: {mse_tau:.6f}")


if __name__ == "__main__":
    main()
