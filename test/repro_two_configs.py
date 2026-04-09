import argparse
import os
import sys
from typing import Any, Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ate import tau_vector_and_se_from_gnn
from src.gen_data import sample_data_spillover

ESTIMANDS = ("tau_dir", "tau_in", "tau_out", "tau_tot")


def _run_fit(
    draw: Dict[str, Any],
    directed: bool,
    variance_type: str,
    variance_method: Optional[str],
    num_layers: int,
    output_dim: int,
    seed: int,
    use_gpu: bool,
) -> Dict[str, Any]:
    return tau_vector_and_se_from_gnn(
        draw,
        feature_key="node_features",
        directed=directed,
        use_gpu=use_gpu,
        variance_type=variance_type,
        variance_method=variance_method,
        num_layers=num_layers,
        output_dim=output_dim,
        seed=seed,
    )


def _max_abs_diff(a: Dict[str, float], b: Dict[str, float]) -> float:
    return max(abs(float(a[k]) - float(b[k])) for k in ESTIMANDS)


def _fmt_map(values: Dict[str, float]) -> str:
    return ", ".join(f"{k}={float(values[k]):.6f}" for k in ESTIMANDS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Quick reproducibility check on identical spillover data for two estimator configs. "
            "Data generation follows test/test_spillover_pipeline.py via sample_data_spillover()."
        )
    )
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed_data", type=int, default=11)
    parser.add_argument("--graph_model", type=str, default="rgg", choices=("rgg", "er"))
    parser.add_argument("--tau_dir_true", type=float, default=2.0)
    parser.add_argument("--tau_in_true", type=float, default=1.0)
    parser.add_argument("--tau_out_true", type=float, default=1.0)

    parser.add_argument("--directed", action="store_true", help="Use directed nuisance fits (DirGNN).")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=4)
    parser.add_argument("--seed_model", type=int, default=7)
    parser.add_argument("--use_gpu", action="store_true", help="Allow CUDA if available.")

    parser.add_argument("--cfg_a_variance_type", type=str, default="directed")
    parser.add_argument("--cfg_a_variance_method", type=str, default="dir_avg")
    parser.add_argument("--cfg_b_variance_type", type=str, default="iid")
    parser.add_argument("--cfg_b_variance_method", type=str, default="")

    parser.add_argument(
        "--tau_tol",
        type=float,
        default=1e-6,
        help="Fail if max |tau_hat(A)-tau_hat(B)| exceeds this tolerance.",
    )
    parser.add_argument(
        "--determinism_tol",
        type=float,
        default=1e-8,
        help="Fail if repeating config A changes tau_hat by more than this tolerance.",
    )
    args = parser.parse_args()

    draw = sample_data_spillover(
        sample_size=args.n,
        seed=args.seed_data,
        graph_model=args.graph_model,
        tau_dir=args.tau_dir_true,
        tau_in=args.tau_in_true,
        tau_out=args.tau_out_true,
    )

    a_method = args.cfg_a_variance_method.strip() or None
    b_method = args.cfg_b_variance_method.strip() or None

    fit_a = _run_fit(
        draw,
        directed=bool(args.directed),
        variance_type=args.cfg_a_variance_type,
        variance_method=a_method,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        seed=args.seed_model,
        use_gpu=bool(args.use_gpu),
    )
    fit_b = _run_fit(
        draw,
        directed=bool(args.directed),
        variance_type=args.cfg_b_variance_type,
        variance_method=b_method,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        seed=args.seed_model,
        use_gpu=bool(args.use_gpu),
    )
    fit_a_repeat = _run_fit(
        draw,
        directed=bool(args.directed),
        variance_type=args.cfg_a_variance_type,
        variance_method=a_method,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        seed=args.seed_model,
        use_gpu=bool(args.use_gpu),
    )

    tau_diff_ab = _max_abs_diff(fit_a["tau_hat"], fit_b["tau_hat"])
    tau_diff_aa = _max_abs_diff(fit_a["tau_hat"], fit_a_repeat["tau_hat"])
    se_diff_ab = _max_abs_diff(fit_a["se_hat"], fit_b["se_hat"])

    print("=== Reproducibility Check (same generated data, two configs) ===")
    print(
        f"data: n={args.n}, seed_data={args.seed_data}, graph_model={args.graph_model}, "
        f"true_taus={draw['true_taus']}"
    )
    print(
        "model: "
        f"directed={bool(args.directed)}, num_layers={args.num_layers}, "
        f"output_dim={args.output_dim}, seed_model={args.seed_model}, use_gpu={bool(args.use_gpu)}"
    )
    print(
        "config A: "
        f"variance_type={args.cfg_a_variance_type}, variance_method={fit_a['variance_method']['tau_dir']}"
    )
    print(
        "config B: "
        f"variance_type={args.cfg_b_variance_type}, variance_method={fit_b['variance_method']['tau_dir']}"
    )
    print()

    print("tau_hat(A):", _fmt_map(fit_a["tau_hat"]))
    print("tau_hat(B):", _fmt_map(fit_b["tau_hat"]))
    print("se_hat(A): ", _fmt_map(fit_a["se_hat"]))
    print("se_hat(B): ", _fmt_map(fit_b["se_hat"]))
    print()

    print(f"max |tau_hat(A)-tau_hat(B)|       = {tau_diff_ab:.10f}")
    print(f"max |tau_hat(A)-tau_hat(A_repeat)| = {tau_diff_aa:.10f}")
    print(f"max |se_hat(A)-se_hat(B)|          = {se_diff_ab:.10f}")

    ok_tau_ab = tau_diff_ab <= float(args.tau_tol)
    ok_tau_aa = tau_diff_aa <= float(args.determinism_tol)

    print()
    print(f"tau invariance across configs (tol={args.tau_tol}): {ok_tau_ab}")
    print(f"determinism within config A (tol={args.determinism_tol}): {ok_tau_aa}")

    if ok_tau_ab and ok_tau_aa:
        print("PASS")
        return 0

    print("FAIL")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise
