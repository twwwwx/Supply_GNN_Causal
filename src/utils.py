from __future__ import annotations

import numpy as np


def to_1d_float(values) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return [float(arr)]
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        raise ValueError("Expected a 1D input.")
    return arr.astype(float).tolist()


def to_2d_float(features) -> list[list[float]]:
    arr = np.asarray(features, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError("Expected 1D or 2D features.")
    return arr.astype(float).tolist()


def clip_probs(p: list[float], clip: float = 1e-3) -> list[float]:
    lo = clip
    hi = 1.0 - clip
    return [min(max(v, lo), hi) for v in p]

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
