#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dGC.baseline import estimate_tau_hat_dr_linear
from dGC.gen_data import sample_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Monte Carlo MSE of tau_hat (LR DR estimator).")
    parser.add_argument("--num-preds", type=int, default=100, help="Number of Monte Carlo predictions.")
    parser.add_argument("--sample-size", type=int, default=100, help="Sample size per draw.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--graph-model", type=str, default="rgg", choices=["rgg", "er"], help="Graph model.")
    parser.add_argument("--feature-key", type=str, default="X", help="Feature key for the estimator.")
    parser.add_argument("--true-tau", type=float, default=0.0, help="Reference true tau for MSE.")
    args = parser.parse_args()

    tau_hats = []
    for i in range(args.num_preds):
        draw = sample_data(
            sample_size=args.sample_size,
            seed=args.seed + i,
            graph_model=args.graph_model,
        )
        fit = estimate_tau_hat_dr_linear(draw, feature_key=args.feature_key)
        tau_hats.append(float(fit["tau_hat"]))

    tau_arr = np.asarray(tau_hats, dtype=float)
    mse_tau = float(np.mean((tau_arr - float(args.true_tau)) ** 2))
    print(f"MSE(tau_hat) over {args.num_preds} predictions: {mse_tau:.6f}")


if __name__ == "__main__":
    main()
