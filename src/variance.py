from __future__ import annotations

import math

import igraph as ig
import numpy as np


def _validate_square_adjacency(adjacency: np.ndarray) -> np.ndarray:
    a = np.asarray(adjacency)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    return (a != 0).astype(np.int8)


def _symmetrize(adjacency: np.ndarray) -> np.ndarray:
    a = _validate_square_adjacency(adjacency)
    return np.logical_or(a != 0, a.T != 0).astype(np.int8)


def _all_pairs_shortest_paths(adjacency: np.ndarray) -> np.ndarray:
    """Unweighted all-pairs shortest paths via igraph distances."""
    a = _validate_square_adjacency(adjacency)
    g = ig.Graph.Adjacency(a.tolist(), mode="directed")
    return np.asarray(g.distances(mode="out"), dtype=float)


def select_bandwidth(adjacency: np.ndarray, directed: bool = False) -> int:
    """Bandwidth rule inspired by the main.tex note on path length / average degree."""
    a = _validate_square_adjacency(adjacency)
    graph = a if directed else _symmetrize(a)
    dist = _all_pairs_shortest_paths(graph)
    n = int(graph.shape[0])

    finite = dist[np.isfinite(dist) & (dist > 0.0)]
    if finite.size == 0:
        return 1
    avg_path_len = float(np.mean(finite))

    avg_degree = float(np.mean(np.sum(graph, axis=1)))
    if avg_degree <= 1.0:
        return max(1, int(math.ceil(avg_path_len ** 0.25)))

    denom = math.log(max(avg_degree, 1.0000001))
    threshold = 2.0 * math.log(max(n, 2)) / denom if denom > 0.0 else float("inf")
    if avg_path_len < threshold:
        b = int(math.ceil(0.25 * avg_path_len))
    else:
        b = int(math.ceil(avg_path_len ** 0.25))
    return max(1, b)


def _u_kernel_from_dist(dist: np.ndarray, bandwidth: int) -> np.ndarray:
    b = int(bandwidth)
    return (np.isfinite(dist) & (dist <= b)).astype(float)


def _pd_kernel_from_mask(mask: np.ndarray) -> np.ndarray:
    counts = np.sum(mask, axis=1).astype(float)
    counts = np.where(counts > 0.0, counts, 1.0)
    inter = mask.astype(float) @ mask.astype(float).T
    denom = np.sqrt(np.outer(counts, counts))
    return inter / denom


def _variance_from_kernel(tau_tilde: np.ndarray, kernel: np.ndarray, m_n: float | None = None) -> float:
    t = np.asarray(tau_tilde, dtype=float).reshape(-1)
    if kernel.shape != (t.size, t.size):
        raise ValueError("kernel shape must be (n, n) with n=len(tau_tilde).")
    if m_n is None:
        m_n = float(t.size)
    if m_n <= 0.0:
        return 0.0
    return float((t @ kernel @ t) / m_n)


def estimate_variance_skeleton(
    tau_tilde: np.ndarray,
    adjacency_directed: np.ndarray,
    bandwidth: int | None = None,
    method: str = "max",
) -> dict:
    """Undirected-skeleton estimators: choose method in {'u','pd','max'}."""
    method_norm = method.lower()
    if method_norm not in {"u", "pd", "max"}:
        raise ValueError("method must be one of: 'u', 'pd', 'max'.")

    a_sym = _symmetrize(adjacency_directed)
    b = select_bandwidth(a_sym, directed=False) if bandwidth is None else int(bandwidth)
    dist = _all_pairs_shortest_paths(a_sym)
    m_n = float(np.asarray(tau_tilde).reshape(-1).size)

    k_u = _u_kernel_from_dist(dist, b)
    half_b = max(1, int(math.floor(b / 2)))
    nh_mask = dist <= half_b
    k_pd = _pd_kernel_from_mask(nh_mask)
    k_max = np.maximum(k_u, k_pd)

    sigma2_u = _variance_from_kernel(tau_tilde, k_u, m_n=m_n)
    sigma2_pd = _variance_from_kernel(tau_tilde, k_pd, m_n=m_n)
    sigma2_max = _variance_from_kernel(tau_tilde, k_max, m_n=m_n)
    sigma2_map = {"u": sigma2_u, "pd": sigma2_pd, "max": sigma2_max}
    sigma2 = float(sigma2_map[method_norm])
    se = float(math.sqrt(max(sigma2, 0.0) / m_n)) if m_n > 0 else 0.0

    return {
        "method": method_norm,
        "bandwidth": int(b),
        "m_n": m_n,
        "K_u": k_u,
        "K_pd": k_pd,
        "K_max": k_max,
        "sigma2_u": sigma2_u,
        "sigma2_pd": sigma2_pd,
        "sigma2_max": sigma2_max,
        "sigma2": sigma2,
        "se": se,
    }


def estimate_variance_directed(
    tau_tilde: np.ndarray,
    adjacency_directed: np.ndarray,
    bandwidth: int | None = None,
    method: str = "dir_max",
) -> dict:
    """Directed-neighborhood estimators.

    method in {'in_max', 'out_max', 'dir_max', 'dir_avg'}.
    """
    method_norm = method.lower()
    if method_norm not in {"in_max", "out_max", "dir_max", "dir_avg"}:
        raise ValueError("method must be one of: 'in_max', 'out_max', 'dir_max', 'dir_avg'.")

    a = _validate_square_adjacency(adjacency_directed)
    b = select_bandwidth(a, directed=True) if bandwidth is None else int(bandwidth)
    dist_out = _all_pairs_shortest_paths(a)  # dist_out[i, j] = directed path i -> j
    m_n = float(np.asarray(tau_tilde).reshape(-1).size)

    k_u_out = _u_kernel_from_dist(dist_out, b)  # 1{l(i,j)<=b}
    k_u_in = _u_kernel_from_dist(dist_out.T, b)  # 1{l(j,i)<=b}

    half_b = max(1, int(math.floor(b / 2)))
    out_mask = dist_out <= half_b
    in_mask = dist_out.T <= half_b
    k_pd_out = _pd_kernel_from_mask(out_mask)
    k_pd_in = _pd_kernel_from_mask(in_mask)

    k_out_max = np.maximum(k_u_out, k_pd_out)
    k_in_max = np.maximum(k_u_in, k_pd_in)
    k_dir_max = np.maximum(k_out_max, k_in_max)
    k_dir_avg = 0.5 * (k_out_max + k_in_max)

    sigma2_in_max = _variance_from_kernel(tau_tilde, k_in_max, m_n=m_n)
    sigma2_out_max = _variance_from_kernel(tau_tilde, k_out_max, m_n=m_n)
    sigma2_dir_max = _variance_from_kernel(tau_tilde, k_dir_max, m_n=m_n)
    sigma2_dir_avg = _variance_from_kernel(tau_tilde, k_dir_avg, m_n=m_n)
    sigma2_map = {
        "in_max": sigma2_in_max,
        "out_max": sigma2_out_max,
        "dir_max": sigma2_dir_max,
        "dir_avg": sigma2_dir_avg,
    }
    sigma2 = float(sigma2_map[method_norm])
    se = float(math.sqrt(max(sigma2, 0.0) / m_n)) if m_n > 0 else 0.0

    return {
        "method": method_norm,
        "bandwidth": int(b),
        "m_n": m_n,
        "K_u_in": k_u_in,
        "K_u_out": k_u_out,
        "K_pd_in": k_pd_in,
        "K_pd_out": k_pd_out,
        "K_in_max": k_in_max,
        "K_out_max": k_out_max,
        "K_dir_max": k_dir_max,
        "K_dir_avg": k_dir_avg,
        "sigma2_in_max": sigma2_in_max,
        "sigma2_out_max": sigma2_out_max,
        "sigma2_dir_max": sigma2_dir_max,
        "sigma2_dir_avg": sigma2_dir_avg,
        "sigma2": sigma2,
        "se": se,
    }
