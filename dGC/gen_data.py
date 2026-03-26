from __future__ import annotations

import math
import random
from typing import Literal

X_SUPPORT = [0.0, 0.25, 0.5, 0.75, 1.0]
THETA_D = (-0.5, 1.5, 1.0, -1.0)
THETA_Y = (0.5, 0.8, 10.0, -1.0)


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    out = [0.0] * len(matrix)
    for i, row in enumerate(matrix):
        out[i] = sum(weight * vector[j] for j, weight in enumerate(row))
    return out


def _max_abs_diff(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _row_normalize(adjacency: list[list[int]]) -> tuple[list[list[float]], list[float]]:
    n = len(adjacency)
    degree = [float(sum(adjacency[i])) for i in range(n)]
    weights = [[0.0] * n for _ in range(n)]
    for i in range(n):
        if degree[i] == 0.0:
            continue
        inv_deg = 1.0 / degree[i]
        for j, edge in enumerate(adjacency[i]):
            if edge:
                weights[i][j] = inv_deg
    return weights, degree


def _network_index(
    z: list[float],
    u: list[float],
    x: list[float],
    weights: list[list[float]],
    theta: tuple[float, float, float, float],
) -> list[float]:
    alpha, beta, delta, gamma = theta
    wz = _matvec(weights, z)
    wx = _matvec(weights, x)
    wu = _matvec(weights, u)
    return (
        [
            alpha + beta * wz[i] + delta * wx[i] + gamma * x[i] + u[i] + wu[i]
            for i in range(len(x))
        ]
    )


def _gen_rgg(n: int, rng: random.Random) -> list[list[int]]:
    positions = [(rng.random(), rng.random()) for _ in range(n)]
    r_n = math.sqrt(5.0 / (math.pi * n))
    threshold = r_n * r_n
    adjacency = [[0] * n for _ in range(n)]
    for i in range(n):
        x_i, y_i = positions[i]
        for j in range(i + 1, n):
            x_j, y_j = positions[j]
            dist_sq = (x_i - x_j) ** 2 + (y_i - y_j) ** 2
            if dist_sq <= threshold:
                adjacency[i][j] = 1
                adjacency[j][i] = 1
    return adjacency


def _gen_er(n: int, rng: random.Random) -> list[list[int]]:
    p_n = min(5.0 / n, 1.0)
    adjacency = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_n:
                adjacency[i][j] = 1
                adjacency[j][i] = 1
    return adjacency


def _solve_treatment_equilibrium(
    x: list[float],
    nu: list[float],
    weights: list[list[float]],
    max_iter: int,
) -> tuple[list[int], int]:
    z0 = [0.0] * len(x)
    d_curr = [1 if val > 0.0 else 0 for val in _network_index(z0, nu, x, weights, THETA_D)]
    for step in range(1, max_iter + 1):
        d_float = [float(v) for v in d_curr]
        d_next = [
            1 if val > 0.0 else 0
            for val in _network_index(d_float, nu, x, weights, THETA_D)
        ]
        if d_next == d_curr:
            return d_next, step
        d_curr = d_next
    return d_curr, max_iter


def _solve_outcome_equilibrium(
    x: list[float],
    eps: list[float],
    weights: list[list[float]],
    max_iter: int,
    tol: float,
) -> tuple[list[float], int]:
    wx = _matvec(weights, x)
    weps = _matvec(weights, eps)
    exogenous = [
        THETA_Y[0] + THETA_Y[2] * wx[i] + THETA_Y[3] * x[i] + eps[i] + weps[i]
        for i in range(len(x))
    ]
    y_curr = list(exogenous)
    for step in range(1, max_iter + 1):
        wy = _matvec(weights, y_curr)
        y_next = [exogenous[i] + THETA_Y[1] * wy[i] for i in range(len(x))]
        if _max_abs_diff(y_next, y_curr) <= tol:
            return y_next, step
        y_curr = y_next
    return y_curr, max_iter


def sample_data(
    sample_size: int,
    seed: int,
    graph_model: Literal["rgg", "er"] = "rgg",
    treatment_max_iter: int = 500,
    outcome_max_iter: int = 2000,
    outcome_tol: float = 1e-10,
) -> dict:
    """
    Sample one baseline simulation draw from Section
    "Reproducing the Baseline Simulation Design".
    Returns all core model inputs/objects for nuisance learning:
    graph, node covariates, shocks, treatment equilibrium, and outcomes.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    rng = random.Random(seed)
    model = graph_model.lower()
    if model == "rgg":
        adjacency = _gen_rgg(sample_size, rng)
    elif model == "er":
        adjacency = _gen_er(sample_size, rng)
    else:
        raise ValueError("graph_model must be either 'rgg' or 'er'.")

    weights, degree = _row_normalize(adjacency)
    x = [X_SUPPORT[rng.randrange(len(X_SUPPORT))] for _ in range(sample_size)]
    eps = [rng.gauss(0.0, 1.0) for _ in range(sample_size)]
    nu = [rng.gauss(0.0, 1.0) for _ in range(sample_size)]

    d0 = [
        1 if val > 0.0 else 0
        for val in _network_index([0.0] * sample_size, nu, x, weights, THETA_D)
    ]
    d, treat_iters = _solve_treatment_equilibrium(x, nu, weights, treatment_max_iter)
    y, outcome_iters = _solve_outcome_equilibrium(x, eps, weights, outcome_max_iter, outcome_tol)
    t = list(d)

    # Feature bundles used by first-stage learners.
    neighbor_x = _matvec(weights, x)
    node_features = [[x[i], degree[i], neighbor_x[i]] for i in range(sample_size)]
    tabular_features = [[x[i], neighbor_x[i], degree[i]] for i in range(sample_size)]

    return {
        "n": sample_size,
        "seed": seed,
        "graph_model": model,
        "adjacency": adjacency,
        "row_normalized_adjacency": weights,
        "degree": degree,
        "X": x,
        "epsilon": eps,
        "nu": nu,
        "D_init": d0,
        "D": d,
        "Y": y,
        "T": t,
        "node_features": node_features,
        "tabular_features": tabular_features,
        "theta_d": list(THETA_D),
        "theta_y": list(THETA_Y),
        "convergence": {
            "treatment_iterations": treat_iters,
            "outcome_iterations": outcome_iters,
            "treatment_converged": treat_iters < treatment_max_iter,
            "outcome_converged": outcome_iters < outcome_max_iter,
        },
        # Backward-compatible view for legacy code expecting (x, y) rows.
        "legacy_rows": list(zip(x, y)),
    }
