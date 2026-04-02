from __future__ import annotations

import math
from typing import Literal

import numpy as np

X_SUPPORT = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
THETA_D = (-0.5, 1.5, 1.0, -1.0)
THETA_Y_MANUAL = (0.5, 2.0, 10.0, 1.0)  # (theta_1, theta_2=tau, theta_3, theta_4)
THETA_D_DIR = (-0.5, 1.0, 0.8, 1.0, -0.6, 1.0)  # (alpha, b_in, b_out, d_in, d_out, gamma)
THETA_Y_DIR = (0.5, 2.0, 8.0, 4.0, 1.0)  # (theta_1, theta_2=tau, theta_in, theta_out, theta_x)


def _matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return matrix @ vector


def _network_index(
    z: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    theta: tuple[float, float, float, float],
) -> np.ndarray:
    alpha, beta, delta, gamma = theta
    return alpha + beta * (weights @ z) + delta * (weights @ x) + gamma * x + u + (weights @ u)


def _network_index_dir(
    z: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    weights_in: np.ndarray,
    weights_out: np.ndarray,
    theta: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    alpha, beta_in, beta_out, delta_in, delta_out, gamma = theta
    return (
        alpha
        + beta_in * (weights_in @ z)
        + beta_out * (weights_out @ z)
        + delta_in * (weights_in @ x)
        + delta_out * (weights_out @ x)
        + gamma * x
        + u
        + (weights_in @ u)
        + (weights_out @ u)
    )


def _solve_treatment_equilibrium(
    x: np.ndarray,
    nu: np.ndarray,
    weights: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, int]:
    d_curr = (_network_index(np.zeros_like(x), nu, x, weights, THETA_D) > 0.0).astype(int)
    for step in range(1, max_iter + 1):
        d_next = (_network_index(d_curr.astype(float), nu, x, weights, THETA_D) > 0.0).astype(int)
        if np.array_equal(d_next, d_curr):
            return d_next, step
        d_curr = d_next
    return d_curr, max_iter


def _solve_treatment_equilibrium_dir(
    x: np.ndarray,
    nu: np.ndarray,
    weights_in: np.ndarray,
    weights_out: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, int]:
    d_curr = (
        _network_index_dir(np.zeros_like(x), nu, x, weights_in, weights_out, THETA_D_DIR) > 0.0
    ).astype(int)
    for step in range(1, max_iter + 1):
        d_next = (
            _network_index_dir(d_curr.astype(float), nu, x, weights_in, weights_out, THETA_D_DIR) > 0.0
        ).astype(int)
        if np.array_equal(d_next, d_curr):
            return d_next, step
        d_curr = d_next
    return d_curr, max_iter


def _row_normalize(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    degree = adjacency.sum(axis=1).astype(float)
    denom = np.where(degree > 0.0, degree, 1.0)
    weights = adjacency.astype(float) / denom[:, None]
    return weights, degree


def _check_symmetric_adjacency(adjacency: np.ndarray) -> None:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be square.")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("Adjacency must be symmetric.")


def _gen_rgg(n: int, rng: np.random.Generator) -> np.ndarray:
    positions = rng.random((n, 2))
    r_n = math.sqrt(5.0 / (math.pi * n))
    threshold_sq = r_n * r_n
    adjacency = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        diff = positions[i + 1 :] - positions[i]
        dist_sq = np.sum(diff * diff, axis=1)
        idx = np.where(dist_sq <= threshold_sq)[0]
        if idx.size == 0:
            continue
        j = idx + i + 1
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    return adjacency


def _gen_er(n: int, rng: np.random.Generator) -> np.ndarray:
    p_n = min(5.0 / n, 1.0)
    upper = rng.random((n, n))
    adjacency = np.triu((upper < p_n).astype(np.int8), 1)
    adjacency = adjacency + adjacency.T
    return adjacency


def _orient_undirected_edges(skeleton: np.ndarray, rng: np.random.Generator, p_bidirected: float) -> np.ndarray:
    if not (0.0 <= p_bidirected < 1.0):
        raise ValueError("p_bidirected must be in [0, 1).")
    directed = np.zeros_like(skeleton, dtype=np.int8)
    edges = np.argwhere(np.triu(skeleton, 1) > 0)
    if edges.size == 0:
        return directed
    draws = rng.random(edges.shape[0])
    p_forward = (1.0 - p_bidirected) / 2.0
    for idx, (i, j) in enumerate(edges):
        if draws[idx] < p_bidirected:
            directed[i, j] = 1
            directed[j, i] = 1
        elif draws[idx] < p_bidirected + p_forward:
            directed[i, j] = 1
        else:
            directed[j, i] = 1
    return directed


def _gen_rgg_dir(n: int, rng: np.random.Generator, p_bidirected: float) -> np.ndarray:
    return _orient_undirected_edges(_gen_rgg(n, rng), rng, p_bidirected)


def _gen_er_dir(n: int, rng: np.random.Generator, p_bidirected: float) -> np.ndarray:
    return _orient_undirected_edges(_gen_er(n, rng), rng, p_bidirected)


def gen_heterophilic_labels(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    support: np.ndarray = X_SUPPORT,
) -> np.ndarray:
    """Sample node labels by pairing connected nodes with different values when possible."""
    a = np.asarray(adjacency)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Adjacency must be square.")

    support = np.asarray(support, dtype=float)
    if np.unique(support).size < 2:
        raise ValueError("support must contain at least two distinct values.")

    n = a.shape[0]
    x = np.empty(n, dtype=float)
    assigned = np.zeros(n, dtype=bool)

    skeleton = np.logical_or(a != 0, a.T != 0)
    np.fill_diagonal(skeleton, False)
    pairs = np.argwhere(np.triu(skeleton, 1) > 0).astype(np.int64)
    if pairs.size > 0:
        rng.shuffle(pairs, axis=0)
        for i, j in pairs:
            if assigned[i] or assigned[j]:
                continue
            pair_vals = rng.choice(support, size=2, replace=False)
            x[i] = pair_vals[0]
            x[j] = pair_vals[1]
            assigned[i] = True
            assigned[j] = True

    remaining = np.where(~assigned)[0]
    if remaining.size > 0:
        x[remaining] = rng.choice(support, size=remaining.size, replace=True)

    return x


def sample_data_undir(
    sample_size: int,
    seed: int,
    graph_model: Literal["rgg", "er"] = "rgg",
    treatment_max_iter: int = 500,
    outcome_max_iter: int = 2000,
    outcome_tol: float = 1e-10,
) -> dict:
    del outcome_max_iter, outcome_tol
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    rng = np.random.default_rng(seed)
    model = graph_model.lower()
    if model == "rgg":
        adjacency = _gen_rgg(sample_size, rng)
    elif model == "er":
        adjacency = _gen_er(sample_size, rng)
    else:
        raise ValueError("graph_model must be either 'rgg' or 'er'.")
    _check_symmetric_adjacency(adjacency)

    weights, degree = _row_normalize(adjacency)
    x = gen_heterophilic_labels(adjacency, rng)
    nu = rng.normal(0.0, 1.0, size=sample_size)
    u = rng.normal(0.0, 1.0, size=sample_size)
    d0 = (_network_index(np.zeros(sample_size, dtype=float), nu, x, weights, THETA_D) > 0.0).astype(int)
    d, treat_iters = _solve_treatment_equilibrium(x, nu, weights, treatment_max_iter)

    neighbor_x = _matvec(weights, x)
    wu = _matvec(weights, u)
    theta_1, theta_2, theta_3, theta_4 = THETA_Y_MANUAL
    y = theta_1 + theta_2 * d + theta_3 * neighbor_x - theta_4 * x + u + wu

    node_features = np.column_stack([x, degree, neighbor_x]).astype(float)
    tabular_features = np.column_stack([x, neighbor_x, degree]).astype(float)

    return {
        "n": int(sample_size),
        "seed": int(seed),
        "graph_model": model,
        "dgp_variant": "undir",
        "true_tau": float(theta_2),
        "adjacency": adjacency,
        "row_normalized_adjacency": weights,
        "degree": degree,
        "X": x,
        "epsilon": u,
        "nu": nu,
        "D_init": d0,
        "D": d,
        "Y": y,
        "T": d.copy(),
        "node_features": node_features,
        "tabular_features": tabular_features,
        "theta_d": np.asarray(THETA_D, dtype=float),
        "theta_y": np.asarray(THETA_Y_MANUAL, dtype=float),
        "convergence": {
            "treatment_iterations": int(treat_iters),
            "outcome_iterations": 1,
            "treatment_converged": bool(treat_iters < treatment_max_iter),
            "outcome_converged": True,
        },
        "legacy_rows": np.column_stack([x, y]),
    }


def sample_data_dir(
    sample_size: int,
    seed: int,
    graph_model: Literal["rgg", "er"] = "rgg",
    p_bidirected: float = 0.05,
    treatment_max_iter: int = 500,
    outcome_max_iter: int = 2000,
    outcome_tol: float = 1e-10,
) -> dict:
    del outcome_max_iter, outcome_tol
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    rng = np.random.default_rng(seed)
    model = graph_model.lower()
    if model == "rgg":
        adjacency = _gen_rgg_dir(sample_size, rng, p_bidirected=p_bidirected)
    elif model == "er":
        adjacency = _gen_er_dir(sample_size, rng, p_bidirected=p_bidirected)
    else:
        raise ValueError("graph_model must be either 'rgg' or 'er'.")

    weights_out, out_degree = _row_normalize(adjacency)
    weights_in, in_degree = _row_normalize(adjacency.T)

    x = gen_heterophilic_labels(adjacency, rng)
    nu = rng.normal(0.0, 1.0, size=sample_size)
    u = rng.normal(0.0, 1.0, size=sample_size)
    d0 = (
        _network_index_dir(
            np.zeros(sample_size, dtype=float),
            nu,
            x,
            weights_in,
            weights_out,
            THETA_D_DIR,
        )
        > 0.0
    ).astype(int)
    d, treat_iters = _solve_treatment_equilibrium_dir(
        x=x,
        nu=nu,
        weights_in=weights_in,
        weights_out=weights_out,
        max_iter=treatment_max_iter,
    )

    neighbor_x_in = _matvec(weights_in, x)
    neighbor_x_out = _matvec(weights_out, x)
    wu_in = _matvec(weights_in, u)
    wu_out = _matvec(weights_out, u)
    theta_1, theta_2, theta_in, theta_out, theta_x = THETA_Y_DIR
    y = theta_1 + theta_2 * d + theta_in * neighbor_x_in + theta_out * neighbor_x_out - theta_x * x + u + wu_in + wu_out

    degree_total = in_degree + out_degree
    node_features = np.column_stack([x, in_degree, out_degree, neighbor_x_in, neighbor_x_out]).astype(float)
    tabular_features = np.column_stack([x, neighbor_x_in, neighbor_x_out, in_degree, out_degree]).astype(float)

    return {
        "n": int(sample_size),
        "seed": int(seed),
        "graph_model": model,
        "dgp_variant": "dir",
        "true_tau": float(theta_2),
        "adjacency": adjacency,
        "adjacency_skeleton": np.logical_or(adjacency != 0, adjacency.T != 0).astype(np.int8),
        "row_normalized_adjacency": weights_out,
        "row_normalized_adjacency_in": weights_in,
        "row_normalized_adjacency_out": weights_out,
        "degree": out_degree,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "degree_total": degree_total,
        "X": x,
        "epsilon": u,
        "nu": nu,
        "D_init": d0,
        "D": d,
        "Y": y,
        "T": d.copy(),
        "node_features": node_features,
        "tabular_features": tabular_features,
        "theta_d": np.asarray(THETA_D_DIR, dtype=float),
        "theta_y": np.asarray(THETA_Y_DIR, dtype=float),
        "convergence": {
            "treatment_iterations": int(treat_iters),
            "outcome_iterations": 1,
            "treatment_converged": bool(treat_iters < treatment_max_iter),
            "outcome_converged": True,
        },
        "legacy_rows": np.column_stack([x, y]),
    }


# Backward-compatible alias for existing callers.
sample_data = sample_data_undir


def sample_data_simple(
    sample_size: int,
    seed: int,
    graph_model: Literal["rgg", "er"] = "rgg",
    tau: float = 2.0,
    p_treat: float = 0.5,
) -> dict:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if not (0.0 <= p_treat <= 1.0):
        raise ValueError("p_treat must be in [0, 1].")

    rng = np.random.default_rng(seed)
    model = graph_model.lower()
    if model == "rgg":
        adjacency = _gen_rgg(sample_size, rng)
    elif model == "er":
        adjacency = _gen_er(sample_size, rng)
    else:
        raise ValueError("graph_model must be either 'rgg' or 'er'.")
    _check_symmetric_adjacency(adjacency)

    weights, degree = _row_normalize(adjacency)
    x = gen_heterophilic_labels(adjacency, rng)
    eps = rng.normal(0.0, 1.0, size=sample_size)
    d = rng.binomial(1, p_treat, size=sample_size).astype(int)  # Bernoulli i.i.d.

    neighbor_x = _matvec(weights, x)
    neighbor_eps = _matvec(weights, eps)
    y = 10.0 * neighbor_x - x + eps + neighbor_eps + float(tau) * d

    node_features = np.column_stack([x, degree, neighbor_x]).astype(float)
    tabular_features = np.column_stack([x, neighbor_x, degree]).astype(float)

    return {
        "n": int(sample_size),
        "seed": int(seed),
        "graph_model": model,
        "dgp_variant": "simple_undir",
        "true_tau": float(tau),
        "adjacency": adjacency,
        "row_normalized_adjacency": weights,
        "degree": degree,
        "X": x,
        "epsilon": eps,
        "nu": np.zeros(sample_size, dtype=float),
        "D_init": d.copy(),
        "D": d,
        "Y": y,
        "T": d.copy(),
        "node_features": node_features,
        "tabular_features": tabular_features,
        "theta_d": None,
        "theta_y": np.asarray([0.0, 0.0, 10.0, -1.0], dtype=float),
        "convergence": {
            "treatment_iterations": 1,
            "outcome_iterations": 1,
            "treatment_converged": True,
            "outcome_converged": True,
        },
        "legacy_rows": np.column_stack([x, y]),
    }


if __name__ == "__main__":
    draw = sample_data_dir(sample_size=2000, seed=123, graph_model="rgg")
    d = np.asarray(draw["D"], dtype=float)
    w_in = np.asarray(draw["row_normalized_adjacency_in"], dtype=float)
    w_out = np.asarray(draw["row_normalized_adjacency_out"], dtype=float)

    treated_prop = float(np.mean(d))
    treated_prop_in_nodes = w_in @ d
    treated_prop_out_nodes = w_out @ d

    print(f"treated_proportion={treated_prop:.6f}")
    print(f"in_nodes_treated_prop_mean={float(np.mean(treated_prop_in_nodes)):.6f}")
    print(f"in_nodes_treated_prop_median={float(np.median(treated_prop_in_nodes)):.6f}")
    print(f"out_nodes_treated_prop_mean={float(np.mean(treated_prop_out_nodes)):.6f}")
    print(f"out_nodes_treated_prop_median={float(np.median(treated_prop_out_nodes)):.6f}")
    print(f"prop_rho_in_eq_1={float(np.mean(treated_prop_in_nodes == 1.0)):.6f}")
    print(f"prop_rho_in_eq_0={float(np.mean(treated_prop_in_nodes == 0.0)):.6f}")
    print(f"prop_rho_out_eq_1={float(np.mean(treated_prop_out_nodes == 1.0)):.6f}")
    print(f"prop_rho_out_eq_0={float(np.mean(treated_prop_out_nodes == 0.0)):.6f}")
