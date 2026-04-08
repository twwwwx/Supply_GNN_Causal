from __future__ import annotations

import warnings

import numpy as np

try:
    from .GNN import (
        GNN_reg_dir_multiclass,
        GNN_reg_dir_outcome_surface,
        GNN_reg_multiclass,
        GNN_reg_outcome_surface,
    )
    from .variance import estimate_variance_directed, estimate_variance_skeleton
except ImportError:
    from GNN import (
        GNN_reg_dir_multiclass,
        GNN_reg_dir_outcome_surface,
        GNN_reg_multiclass,
        GNN_reg_outcome_surface,
    )
    from variance import estimate_variance_directed, estimate_variance_skeleton


EXPOSURE_STATES: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
]
STATE_TO_INDEX: dict[tuple[int, int, int], int] = {state: idx for idx, state in enumerate(EXPOSURE_STATES)}
ESTIMAND_TARGETS: dict[str, tuple[int, int, int]] = {
    "tau_dir": (1, 0, 0),
    "tau_in": (0, 1, 0),
    "tau_out": (0, 0, 1),
    "tau_tot": (1, 1, 1),
}
BASELINE_STATE: tuple[int, int, int] = (0, 0, 0)


def _exposure_matrix_and_state_index(data: dict, exposure_key: str = "T") -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(data[exposure_key])
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError(
            "Expected exposure matrix `T` with shape (n, 3): columns (D, rho_in, rho_out)."
        )
    t_bin = (t > 0.5).astype(int)
    if np.any((t_bin < 0) | (t_bin > 1)):
        raise ValueError("Exposure values must be binary.")
    state_index = (4 * t_bin[:, 0] + 2 * t_bin[:, 1] + t_bin[:, 2]).astype(int)  # Encode (D, rho_in, rho_out) as 3-bit class in {0,...,7}.
    return t_bin, state_index


def _clip_and_renormalize_multiclass_probs(probs: np.ndarray, clip: float) -> np.ndarray:
    p = np.asarray(probs, dtype=float)

    p = np.maximum(p, clip)
    row_sum = np.sum(p, axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
    return p / row_sum


def _contrast_scores(
    y: np.ndarray,
    observed_state_idx: np.ndarray,
    mu_hat: np.ndarray,
    e_hat: np.ndarray,
    target_idx: int,
    baseline_idx: int,
) -> dict:
    mu_t = np.asarray(mu_hat[:, target_idx], dtype=float).reshape(-1)
    mu_0 = np.asarray(mu_hat[:, baseline_idx], dtype=float).reshape(-1)
    e_t = np.asarray(e_hat[:, target_idx], dtype=float).reshape(-1)
    e_0 = np.asarray(e_hat[:, baseline_idx], dtype=float).reshape(-1)
    ind_t = (observed_state_idx == int(target_idx)).astype(float)
    ind_0 = (observed_state_idx == int(baseline_idx)).astype(float)

    psi = ind_t * (y - mu_t) / e_t + mu_t - ind_0 * (y - mu_0) / e_0 - mu_0
    tau_hat = float(np.mean(psi))
    tau_tilde = psi - mu_t + mu_0
    return {
        "psi": psi,
        "tau_hat": tau_hat,
        "tau_tilde": tau_tilde,
    }


def tau_vector_and_se_from_gnn(
    data: dict,
    outcome_key: str = "Y",
    feature_key: str = "node_features",
    exposure_key: str = "T",
    clip: float = 1e-3,
    num_layers: int = 2,
    output_dim: int = 6,
    seed: int = 0,
    directed: bool = False,
    use_gpu: bool = True,
    variance_type: str = "skeleton",
    variance_method: str | None = None,
    bandwidth: int | None = None,
) -> dict:
    y = np.asarray(data[outcome_key], dtype=float).reshape(-1)
    x = np.asarray(data[feature_key], dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    a = np.asarray(data["adjacency"])
    t_obs, state_obs = _exposure_matrix_and_state_index(data, exposure_key=exposure_key)
    n = int(y.size)
    if t_obs.shape[0] != n or x.shape[0] != n or a.shape[0] != n:
        raise ValueError("Y, X, T, and adjacency dimensions are inconsistent.")

    prop_fn = GNN_reg_dir_multiclass if directed else GNN_reg_multiclass
    out_fn = GNN_reg_dir_outcome_surface if directed else GNN_reg_outcome_surface

    e_hat = np.asarray(
        prop_fn(
            labels=state_obs,
            X=x,
            A=a,
            num_classes=8,
            num_layers=num_layers,
            output_dim=output_dim,
            seed=seed,
            use_gpu=use_gpu,
        ),
        dtype=float,
    )
    e_hat = _clip_and_renormalize_multiclass_probs(e_hat, clip=clip)

    mu_hat = np.asarray(
        out_fn(
            Y=y,
            X=x,
            A=a,
            exposure_obs=t_obs,
            states=EXPOSURE_STATES,
            num_layers=num_layers,
            output_dim=output_dim,
            seed=seed + 1,
            use_gpu=use_gpu,
        ),
        dtype=float,
    )
    if mu_hat.shape != (n, len(EXPOSURE_STATES)):
        raise ValueError("Outcome response surface must have shape (n, 8).")

    vtype = variance_type.lower()
    if (not directed) and vtype == "directed":
        warnings.warn(
            "variance_type='directed' requested with undirected nuisance fits "
            "(directed=False). Falling back to variance_type='skeleton'.",
            UserWarning,
        )
        vtype = "skeleton"
        if variance_method is None or str(variance_method).startswith("dir"):
            variance_method = "max"

    tau_hat_map: dict[str, float] = {}
    se_hat_map: dict[str, float] = {}
    sigma2_hat_map: dict[str, float] = {}
    bandwidth_map: dict[str, int | None] = {}
    variance_method_map: dict[str, str] = {}
    psi_map: dict[str, np.ndarray] = {}

    baseline_idx = int(STATE_TO_INDEX[BASELINE_STATE])
    for name, target_state in ESTIMAND_TARGETS.items():
        target_idx = int(STATE_TO_INDEX[target_state])
        contrast = _contrast_scores(
            y=y,
            observed_state_idx=state_obs,
            mu_hat=mu_hat,
            e_hat=e_hat,
            target_idx=target_idx,
            baseline_idx=baseline_idx,
        )
        psi = np.asarray(contrast["psi"], dtype=float)
        tau_tilde = np.asarray(contrast["tau_tilde"], dtype=float)
        tau_hat_map[name] = float(contrast["tau_hat"])
        psi_map[name] = psi

        if vtype == "iid":
            n_float = float(psi.size)
            se = float(np.std(psi, ddof=1) / np.sqrt(n_float)) if psi.size > 1 else 0.0
            sigma2 = float(se * se * n_float)
            se_hat_map[name] = se
            sigma2_hat_map[name] = sigma2
            bandwidth_map[name] = None
            variance_method_map[name] = "iid"
            continue

        if vtype == "skeleton":
            vmethod = "max" if variance_method is None else variance_method
            var = estimate_variance_skeleton(
                tau_tilde,
                a,
                bandwidth=bandwidth,
                method=vmethod,
            )
        elif vtype == "directed":
            vmethod = "dir_max" if variance_method is None else variance_method
            var = estimate_variance_directed(
                tau_tilde,
                a,
                bandwidth=bandwidth,
                method=vmethod,
            )
        else:
            raise ValueError("variance_type must be one of: 'iid', 'skeleton', 'directed'.")

        se_hat_map[name] = float(var["se"])
        sigma2_hat_map[name] = float(var["sigma2"])
        bandwidth_map[name] = int(var["bandwidth"])
        variance_method_map[name] = str(var["method"])

    return {
        "tau_hat": tau_hat_map,
        "se_hat": se_hat_map,
        "sigma2_hat": sigma2_hat_map,
        "bandwidth": bandwidth_map,
        "variance_type": vtype,
        "variance_method": variance_method_map,
        "psi": psi_map,
        "states": EXPOSURE_STATES,
        "target_states": ESTIMAND_TARGETS,
        "baseline_state": BASELINE_STATE,
        "propensity": e_hat,
        "mu_hat": mu_hat,
    }


def tau_hat_and_se_from_gnn(*args, **kwargs) -> dict:
    """Backward-compatible wrapper returning the direct effect only."""
    fit = tau_vector_and_se_from_gnn(*args, **kwargs)
    return {
        "tau_hat": float(fit["tau_hat"]["tau_dir"]),
        "se_hat": float(fit["se_hat"]["tau_dir"]),
        "sigma2_hat": float(fit["sigma2_hat"]["tau_dir"]),
        "bandwidth": fit["bandwidth"]["tau_dir"],
        "variance_type": str(fit["variance_type"]),
        "variance_method": str(fit["variance_method"]["tau_dir"]),
    }


def tau_hat_from_gnn(*args, **kwargs) -> float:
    """Backward-compatible wrapper returning the direct effect only."""
    fit = tau_vector_and_se_from_gnn(*args, **kwargs)
    return float(fit["tau_hat"]["tau_dir"])
