from __future__ import annotations

import math
from typing import Literal

import numpy as np

X_SUPPORT = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
THETA_D = (-0.5, 1.5, 1.0, -1.0)
THETA_Y_MANUAL = (0.5, 2.0, 10.0, 1.0)  # (theta_1, theta_2=tau, theta_3, theta_4)


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


def sample_data(
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
    x = rng.choice(X_SUPPORT, size=sample_size, replace=True)
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
        "dgp_variant": "manual_linear_treatment_equilibrium",
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
    x = rng.choice(X_SUPPORT, size=sample_size, replace=True)
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
        "dgp_variant": "simple_iid_bernoulli",
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
    draw = sample_data(sample_size=500, seed=123, graph_model="rgg")
    print(f"treated_proportion={float(np.mean(draw['D'])):.6f}")
