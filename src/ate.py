import numpy as np

try:
    from .GNN import GNN_reg, GNN_reg_dir
    from .utils import clip_probs, doubly_robust_scores
    from .variance import estimate_variance_directed, estimate_variance_skeleton
except ImportError:
    from GNN import GNN_reg, GNN_reg_dir
    from utils import clip_probs, doubly_robust_scores
    from variance import estimate_variance_directed, estimate_variance_skeleton


def _dr_components_from_gnn(
    data: dict,
    outcome_key: str = "Y",
    treatment_key: str = "D",
    feature_key: str = "node_features",
    clip: float = 1e-3,
    num_layers: int = 2,
    output_dim: int = 6,
    seed: int = 0,
    directed: bool = False,
) -> dict:
    y = np.asarray(data[outcome_key], dtype=float).squeeze()
    d = np.asarray(data[treatment_key], dtype=float).squeeze()
    x = np.asarray(data[feature_key], dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    a = data["adjacency"]

    treated_mask = d > 0.5
    control_mask = ~treated_mask
    if not np.any(treated_mask) or not np.any(control_mask):
        raise ValueError("Need both treated and control units to compute DR tau.")
    reg_fn = GNN_reg_dir if directed else GNN_reg

    mu1_hat = np.asarray(
        reg_fn(
            Y=y,
            X=x,
            A=a,
            sample=treated_mask.astype(bool),
            num_layers=num_layers,
            output_dim=output_dim,
            seed=seed,
        ),
        dtype=float,
    )
    mu0_hat = np.asarray(
        reg_fn(
            Y=y,
            X=x,
            A=a,
            sample=control_mask.astype(bool),
            num_layers=num_layers,
            output_dim=output_dim,
            seed=seed + 1,
        ),
        dtype=float,
    )
    p_hat = np.asarray(
        reg_fn(
            Y=d.astype(int),
            X=x,
            A=a,
            num_layers=num_layers,
            output_dim=output_dim,
            seed=seed + 2,
        ),
        dtype=float,
    )
    p_hat = np.asarray(clip_probs(p_hat.tolist(), clip=clip), dtype=float)
    psi = np.asarray(doubly_robust_scores(y, d, mu1_hat, mu0_hat, p_hat), dtype=float)
    tau_hat = float(np.mean(psi))

    return {
        "tau_hat": tau_hat,
        "psi": psi,
        "adjacency": np.asarray(a),
    }


def tau_hat_from_gnn(
    data: dict,
    outcome_key: str = "Y",
    treatment_key: str = "D",
    feature_key: str = "node_features",
    clip: float = 1e-3,
    num_layers: int = 2,
    output_dim: int = 6,
    seed: int = 0,
    directed: bool = False,
) -> float:
    fit = _dr_components_from_gnn(
        data=data,
        outcome_key=outcome_key,
        treatment_key=treatment_key,
        feature_key=feature_key,
        clip=clip,
        num_layers=num_layers,
        output_dim=output_dim,
        seed=seed,
        directed=directed,
    )
    return float(fit["tau_hat"])


def tau_hat_and_se_from_gnn(
    data: dict,
    outcome_key: str = "Y",
    treatment_key: str = "D",
    feature_key: str = "node_features",
    clip: float = 1e-3,
    num_layers: int = 2,
    output_dim: int = 6,
    seed: int = 0,
    directed: bool = False,
    variance_type: str = "skeleton",
    variance_method: str | None = None,
    bandwidth: int | None = None,
) -> dict:
    fit = _dr_components_from_gnn(
        data=data,
        outcome_key=outcome_key,
        treatment_key=treatment_key,
        feature_key=feature_key,
        clip=clip,
        num_layers=num_layers,
        output_dim=output_dim,
        seed=seed,
        directed=directed,
    )

    vtype = variance_type.lower()
    psi = np.asarray(fit["psi"], dtype=float)
    n = float(psi.size)
    if vtype == "iid":
        se = float(np.std(psi, ddof=1) / np.sqrt(n)) if psi.size > 1 else 0.0
        return {
            "tau_hat": float(fit["tau_hat"]),
            "se_hat": se,
            "variance_type": "iid",
            "variance_method": "iid",
            "bandwidth": None,
            "sigma2_hat": float(se * se * n),
        }

    if vtype == "skeleton":
        vmethod = "max" if variance_method is None else variance_method
        var = estimate_variance_skeleton(psi, fit["adjacency"], bandwidth=bandwidth, method=vmethod)
    elif vtype == "directed":
        vmethod = "dir_max" if variance_method is None else variance_method
        var = estimate_variance_directed(psi, fit["adjacency"], bandwidth=bandwidth, method=vmethod)
    else:
        raise ValueError("variance_type must be one of: 'iid', 'skeleton', 'directed'.")

    return {
        "tau_hat": float(fit["tau_hat"]),
        "se_hat": float(var["se"]),
        "variance_type": vtype,
        "variance_method": str(var["method"]),
        "bandwidth": int(var["bandwidth"]),
        "sigma2_hat": float(var["sigma2"]),
    }


if __name__ == "__main__":
    try:
        from .gen_data import sample_data_simple
    except ImportError:
        from gen_data import sample_data_simple

    base_seed = 123
    n = 200
    true_tau = 2.0
    print(f"true_tau={true_tau:.6f}")
    for i in range(10):
        draw = sample_data_simple(
            sample_size=n,
            seed=base_seed + i,
            graph_model="rgg",
            tau=true_tau,
            p_treat=0.5,
        )
        tau_hat = tau_hat_from_gnn(draw, seed=base_seed + i)
        print(f"tau_hat_{i + 1}={tau_hat:.6f}")
