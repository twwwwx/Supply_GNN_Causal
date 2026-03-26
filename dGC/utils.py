from __future__ import annotations

import math
import statistics


def to_1d_float(values) -> list[float]:
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list):
        raise ValueError("Expected a list-like 1D input.")
    if len(values) > 0 and isinstance(values[0], (list, tuple)):
        if all(len(row) == 1 for row in values):
            return [float(row[0]) for row in values]
        raise ValueError("Input appears 2D; expected 1D.")
    return [float(v) for v in values]


def to_2d_float(features) -> list[list[float]]:
    if isinstance(features, tuple):
        features = list(features)
    if not isinstance(features, list):
        raise ValueError("Expected list-like features.")
    if len(features) == 0:
        return []
    if isinstance(features[0], (list, tuple)):
        return [[float(v) for v in row] for row in features]
    return [[float(v)] for v in features]


def add_intercept(X: list[list[float]]) -> list[list[float]]:
    return [[1.0] + row for row in X]


def solve_linear_system(A: list[list[float]], b: list[float]) -> list[float]:
    """
    Solve A x = b with Gauss-Jordan elimination.
    """
    n = len(A)
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Singular system.")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_val = aug[col][col]
        inv_pivot = 1.0 / pivot_val
        for j in range(col, n + 1):
            aug[col][j] *= inv_pivot

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def fit_ols(X: list[list[float]], y: list[float], ridge: float = 1e-8) -> list[float]:
    """
    Closed-form OLS using normal equations with small ridge stabilization.
    """
    n = len(X)
    if n == 0:
        raise ValueError("Cannot fit OLS with empty data.")
    p = len(X[0])
    if len(y) != n:
        raise ValueError("X and y lengths do not match.")

    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        row = X[i]
        yi = y[i]
        for a in range(p):
            Xty[a] += row[a] * yi
            for b in range(p):
                XtX[a][b] += row[a] * row[b]

    for j in range(p):
        XtX[j][j] += ridge

    return solve_linear_system(XtX, Xty)


def predict_linear(X: list[list[float]], beta: list[float]) -> list[float]:
    return [sum(xj * bj for xj, bj in zip(row, beta)) for row in X]


def clip_probs(p: list[float], clip: float = 1e-3) -> list[float]:
    lo = clip
    hi = 1.0 - clip
    return [min(max(v, lo), hi) for v in p]


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def mse(estimates: list[float], true_value: float) -> float:
    return float(statistics.fmean((v - true_value) ** 2 for v in estimates))

def doubly_robust_scores(Y, D, mu1_hat, mu0_hat, p_hat) -> list[float]:
    y = to_1d_float(Y)
    d = [1.0 if float(v) > 0.5 else 0.0 for v in D]
    mu1 = to_1d_float(mu1_hat)
    mu0 = to_1d_float(mu0_hat)
    p = to_1d_float(p_hat)

    n = len(y)
    if len(d) != n or len(mu1) != n or len(mu0) != n or len(p) != n:
        raise ValueError("All DR inputs must have the same length.")

    psi = []
    for yi, di, m1, m0, pi in zip(y, d, mu1, mu0, p):
        psi_i = (m1 - m0) + di * (yi - m1) / pi - (1.0 - di) * (yi - m0) / (1.0 - pi)
        psi.append(psi_i)
    return psi


def standard_error(values) -> float:
    vals = to_1d_float(values)
    n = len(vals)
    if n <= 1:
        return 0.0
    avg = mean(vals)
    var = sum((v - avg) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var / n)
