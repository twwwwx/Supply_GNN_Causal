import numpy as np

try:
    from .GNN import GNN_reg
    from .utils import clip_probs, doubly_robust_scores
except ImportError:
    from GNN import GNN_reg
    from utils import clip_probs, doubly_robust_scores


def tau_hat_from_gnn(
    data: dict,
    outcome_key: str = "Y",
    treatment_key: str = "D",
    feature_key: str = "node_features",
    clip: float = 1e-3,
    num_layers: int = 2,
    output_dim: int = 6,
    seed: int = 0,
) -> float:
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

    mu1_hat = np.asarray(
        GNN_reg(
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
        GNN_reg(
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
        GNN_reg(
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
    return float(np.mean(psi))


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
