#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import PNAConv, SAGEConv
from torch_geometric.utils import degree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ate import (  # noqa: E402
    BASELINE_STATE,
    ESTIMAND_TARGETS,
    EXPOSURE_STATES,
    STATE_TO_INDEX,
    _clip_and_renormalize_multiclass_probs,
    _contrast_scores,
)
from src.GNN import _to_directed_edge_indices  # noqa: E402
from src.gen_data import sample_data_spillover  # noqa: E402
from src.variance import _all_pairs_shortest_paths, _pd_kernel_from_mask, select_bandwidth  # noqa: E402

ESTIMANDS = ("tau_dir", "tau_in", "tau_out", "tau_tot")


def _u_kernel_from_dist(dist: np.ndarray, bandwidth: int) -> np.ndarray:
    b = int(bandwidth)
    return (np.isfinite(dist) & (dist <= b)).astype(float)


def _variance_from_kernel(tau_tilde: np.ndarray, kernel: np.ndarray, m_n: float) -> float:
    t = np.asarray(tau_tilde, dtype=float).reshape(-1)
    return float((t @ kernel @ t) / m_n)


def build_directed_kernels(adjacency: np.ndarray, bandwidth: int | None) -> tuple[int, dict[str, np.ndarray]]:
    a = (np.asarray(adjacency) != 0).astype(np.int8)
    b = select_bandwidth(a, directed=True) if bandwidth is None else int(bandwidth)
    dist_out = _all_pairs_shortest_paths(a)

    k_u_out = _u_kernel_from_dist(dist_out, b)
    k_u_in = _u_kernel_from_dist(dist_out.T, b)

    half_b = max(1, int(math.floor(b / 2)))
    out_mask = dist_out <= half_b
    in_mask = dist_out.T <= half_b
    k_pd_out = _pd_kernel_from_mask(out_mask)
    k_pd_in = _pd_kernel_from_mask(in_mask)
    return b, {
        "k_u_in": k_u_in,
        "k_u_out": k_u_out,
        "k_pd_in": k_pd_in,
        "k_pd_out": k_pd_out,
    }


def sigma2_directed(
    tau_tilde: np.ndarray,
    kernels: dict[str, np.ndarray],
    m_n: float,
    method: str,
) -> float:
    vmethod = method.lower()
    if vmethod not in {"in_max", "out_max", "dir_max", "dir_avg"}:
        raise ValueError("variance_method must be in {in_max,out_max,dir_max,dir_avg}.")

    sigma2_u_in = _variance_from_kernel(tau_tilde, kernels["k_u_in"], m_n)
    sigma2_u_out = _variance_from_kernel(tau_tilde, kernels["k_u_out"], m_n)
    sigma2_pd_in = _variance_from_kernel(tau_tilde, kernels["k_pd_in"], m_n)
    sigma2_pd_out = _variance_from_kernel(tau_tilde, kernels["k_pd_out"], m_n)

    sigma2_in_max = max(sigma2_u_in, sigma2_pd_in)
    sigma2_out_max = max(sigma2_u_out, sigma2_pd_out)
    sigma2_dir_max = max(sigma2_out_max, sigma2_in_max)
    sigma2_dir_avg = 0.5 * (sigma2_out_max + sigma2_in_max)
    sigma2_map = {
        "in_max": sigma2_in_max,
        "out_max": sigma2_out_max,
        "dir_max": sigma2_dir_max,
        "dir_avg": sigma2_dir_avg,
    }
    return float(sigma2_map[vmethod])


class DirectedDualSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, num_layers: int, output_dim: int, target_dim: int, seed: int = 0) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be at least 1.")
        if int(output_dim) < 2:
            raise ValueError("output_dim must be at least 2.")
        torch.manual_seed(int(seed))
        self.num_layers = int(num_layers)
        self.target_dim = int(target_dim)
        self.relu = torch.nn.ReLU()

        in_width = int(output_dim) // 2
        out_width = int(output_dim) - in_width

        self.in_layers = torch.nn.ModuleList()
        self.out_layers = torch.nn.ModuleList()
        self.combine_layers = torch.nn.ModuleList()

        prev = int(in_dim)
        for _ in range(self.num_layers):
            self.in_layers.append(SAGEConv(prev, in_width))
            self.out_layers.append(SAGEConv(prev, out_width))
            self.combine_layers.append(torch.nn.Linear(prev + in_width + out_width, int(output_dim)))
            prev = int(output_dim)

        self.output_layer = torch.nn.Linear(int(output_dim), self.target_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        for i in range(self.num_layers):
            h_in = self.relu(self.in_layers[i](x, data.edge_index_in))
            h_out = self.relu(self.out_layers[i](x, data.edge_index_out))
            x = self.relu(self.combine_layers[i](torch.cat([x, h_in, h_out], dim=-1)))
        out = self.output_layer(x)
        if self.target_dim == 1:
            return torch.squeeze(out)
        return out


class DirectedDualPNA(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        output_dim: int,
        target_dim: int,
        deg_hist_in: torch.Tensor,
        deg_hist_out: torch.Tensor,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be at least 1.")
        if int(output_dim) < 2:
            raise ValueError("output_dim must be at least 2.")
        torch.manual_seed(int(seed))
        self.num_layers = int(num_layers)
        self.target_dim = int(target_dim)
        self.relu = torch.nn.ReLU()

        in_width = int(output_dim) // 2
        out_width = int(output_dim) - in_width
        aggs = ("mean", "sum", "std", "min", "max")
        scalers = ("identity", "amplification", "attenuation")

        self.in_layers = torch.nn.ModuleList()
        self.out_layers = torch.nn.ModuleList()
        self.combine_layers = torch.nn.ModuleList()

        prev = int(in_dim)
        for _ in range(self.num_layers):
            self.in_layers.append(PNAConv(prev, in_width, aggs, scalers, deg=deg_hist_in))
            self.out_layers.append(PNAConv(prev, out_width, aggs, scalers, deg=deg_hist_out))
            self.combine_layers.append(torch.nn.Linear(prev + in_width + out_width, int(output_dim)))
            prev = int(output_dim)

        self.output_layer = torch.nn.Linear(int(output_dim), self.target_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        for i in range(self.num_layers):
            h_in = self.relu(self.in_layers[i](x, data.edge_index_in))
            h_out = self.relu(self.out_layers[i](x, data.edge_index_out))
            x = self.relu(self.combine_layers[i](torch.cat([x, h_in, h_out], dim=-1)))
        out = self.output_layer(x)
        if self.target_dim == 1:
            return torch.squeeze(out)
        return out


def build_deg_hist(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    d = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
    if d.numel() == 0:
        return torch.ones(1, dtype=torch.long)
    return torch.bincount(d, minlength=int(d.max()) + 1)


def fit_model(
    data: Data,
    model: torch.nn.Module,
    criterion,
    optimizer: torch.optim.Optimizer,
    train_mask: torch.Tensor,
    max_iters: int,
    tol: float,
    patience: int,
    min_iters: int,
) -> None:
    prev_loss: float | None = None
    stable_count = 0
    for step in range(int(max_iters)):
        model.train()
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        loss_val = float(loss.detach().cpu().item())
        if prev_loss is not None:
            if abs(prev_loss - loss_val) < float(tol):
                stable_count += 1
            else:
                stable_count = 0
            if (step + 1) >= int(min_iters) and stable_count >= int(patience):
                break
        prev_loss = loss_val


def sage_multiclass_probs(
    labels: np.ndarray,
    x: np.ndarray,
    adjacency: np.ndarray,
    num_layers: int,
    output_dim: int,
    seed: int,
    use_gpu: bool,
    lr: float,
    max_iters: int,
    tol: float,
    patience: int,
    min_iters: int,
) -> np.ndarray:
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    x = np.asarray(x, dtype=float)
    n = int(y.size)
    edge_in, edge_out = _to_directed_edge_indices(adjacency, n=n)

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index_in=torch.tensor(edge_in, dtype=torch.long),
        edge_index_out=torch.tensor(edge_out, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    ).to(device)
    model = DirectedDualSAGE(
        in_dim=int(data.num_node_features),
        num_layers=int(num_layers),
        output_dim=int(output_dim),
        target_dim=8,
        seed=int(seed),
    ).to(device)
    train_mask = torch.ones(n, dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = torch.nn.CrossEntropyLoss()
    fit_model(
        data,
        model,
        criterion,
        optimizer,
        train_mask,
        max_iters=max_iters,
        tol=tol,
        patience=patience,
        min_iters=min_iters,
    )

    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return probs.astype(float)


def sage_outcome_surface(
    y: np.ndarray,
    x: np.ndarray,
    adjacency: np.ndarray,
    exposure_obs: np.ndarray,
    states: list[tuple[int, int, int]],
    num_layers: int,
    output_dim: int,
    seed: int,
    use_gpu: bool,
    lr: float,
    max_iters: int,
    tol: float,
    patience: int,
    min_iters: int,
) -> np.ndarray:
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    exp_obs = np.asarray(exposure_obs, dtype=float)
    n = int(y.size)
    x_train = np.concatenate([x, exp_obs], axis=1)
    edge_in, edge_out = _to_directed_edge_indices(adjacency, n=n)

    data = Data(
        x=torch.tensor(x_train, dtype=torch.float),
        edge_index_in=torch.tensor(edge_in, dtype=torch.long),
        edge_index_out=torch.tensor(edge_out, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.float),
    ).to(device)
    model = DirectedDualSAGE(
        in_dim=int(data.num_node_features),
        num_layers=int(num_layers),
        output_dim=int(output_dim),
        target_dim=1,
        seed=int(seed),
    ).to(device)
    train_mask = torch.ones(n, dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = torch.nn.MSELoss()
    fit_model(
        data,
        model,
        criterion,
        optimizer,
        train_mask,
        max_iters=max_iters,
        tol=tol,
        patience=patience,
        min_iters=min_iters,
    )

    preds = []
    model.eval()
    with torch.no_grad():
        for state in states:
            exp_eval = np.repeat(np.asarray(state, dtype=float)[None, :], repeats=n, axis=0)
            x_eval = np.concatenate([x, exp_eval], axis=1)
            data_eval = Data(
                x=torch.tensor(x_eval, dtype=torch.float, device=device),
                edge_index_in=data.edge_index_in,
                edge_index_out=data.edge_index_out,
                y=data.y,
            )
            pred = model(data_eval).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    return np.column_stack(preds).astype(float)


def pna_multiclass_probs(
    labels: np.ndarray,
    x: np.ndarray,
    adjacency: np.ndarray,
    num_layers: int,
    output_dim: int,
    seed: int,
    use_gpu: bool,
    lr: float,
    max_iters: int,
    tol: float,
    patience: int,
    min_iters: int,
) -> np.ndarray:
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    x = np.asarray(x, dtype=float)
    n = int(y.size)
    edge_in, edge_out = _to_directed_edge_indices(adjacency, n=n)

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index_in=torch.tensor(edge_in, dtype=torch.long),
        edge_index_out=torch.tensor(edge_out, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )
    deg_in = build_deg_hist(data.edge_index_in, num_nodes=n)
    deg_out = build_deg_hist(data.edge_index_out, num_nodes=n)
    data = data.to(device)
    model = DirectedDualPNA(
        in_dim=int(data.num_node_features),
        num_layers=int(num_layers),
        output_dim=int(output_dim),
        target_dim=8,
        deg_hist_in=deg_in,
        deg_hist_out=deg_out,
        seed=int(seed),
    ).to(device)

    train_mask = torch.ones(n, dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = torch.nn.CrossEntropyLoss()
    fit_model(
        data,
        model,
        criterion,
        optimizer,
        train_mask,
        max_iters=max_iters,
        tol=tol,
        patience=patience,
        min_iters=min_iters,
    )

    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return probs.astype(float)


def pna_outcome_surface(
    y: np.ndarray,
    x: np.ndarray,
    adjacency: np.ndarray,
    exposure_obs: np.ndarray,
    states: list[tuple[int, int, int]],
    num_layers: int,
    output_dim: int,
    seed: int,
    use_gpu: bool,
    lr: float,
    max_iters: int,
    tol: float,
    patience: int,
    min_iters: int,
) -> np.ndarray:
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    exp_obs = np.asarray(exposure_obs, dtype=float)
    n = int(y.size)
    x_train = np.concatenate([x, exp_obs], axis=1)
    edge_in, edge_out = _to_directed_edge_indices(adjacency, n=n)

    data = Data(
        x=torch.tensor(x_train, dtype=torch.float),
        edge_index_in=torch.tensor(edge_in, dtype=torch.long),
        edge_index_out=torch.tensor(edge_out, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.float),
    )
    deg_in = build_deg_hist(data.edge_index_in, num_nodes=n)
    deg_out = build_deg_hist(data.edge_index_out, num_nodes=n)
    data = data.to(device)
    model = DirectedDualPNA(
        in_dim=int(data.num_node_features),
        num_layers=int(num_layers),
        output_dim=int(output_dim),
        target_dim=1,
        deg_hist_in=deg_in,
        deg_hist_out=deg_out,
        seed=int(seed),
    ).to(device)

    train_mask = torch.ones(n, dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = torch.nn.MSELoss()
    fit_model(
        data,
        model,
        criterion,
        optimizer,
        train_mask,
        max_iters=max_iters,
        tol=tol,
        patience=patience,
        min_iters=min_iters,
    )

    preds = []
    model.eval()
    with torch.no_grad():
        for state in states:
            exp_eval = np.repeat(np.asarray(state, dtype=float)[None, :], repeats=n, axis=0)
            x_eval = np.concatenate([x, exp_eval], axis=1)
            data_eval = Data(
                x=torch.tensor(x_eval, dtype=torch.float, device=device),
                edge_index_in=data.edge_index_in,
                edge_index_out=data.edge_index_out,
                y=data.y,
            )
            pred = model(data_eval).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    return np.column_stack(preds).astype(float)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    model_name: str,
    y: np.ndarray,
    x: np.ndarray,
    adjacency: np.ndarray,
    t_obs: np.ndarray,
    state_obs: np.ndarray,
    clip: float,
    num_layers: int,
    output_dim: int,
    seed: int,
    use_gpu: bool,
    lr: float,
    max_iters: int,
    tol: float,
    patience: int,
    min_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "dirgnn_pna":
        e_hat_raw = pna_multiclass_probs(
            labels=state_obs,
            x=x,
            adjacency=adjacency,
            num_layers=num_layers,
            output_dim=output_dim,
            seed=seed,
            use_gpu=use_gpu,
            lr=lr,
            max_iters=max_iters,
            tol=tol,
            patience=patience,
            min_iters=min_iters,
        )
        mu_hat = pna_outcome_surface(
            y=y,
            x=x,
            adjacency=adjacency,
            exposure_obs=t_obs,
            states=EXPOSURE_STATES,
            num_layers=num_layers,
            output_dim=output_dim,
            seed=seed + 1,
            use_gpu=use_gpu,
            lr=lr,
            max_iters=max_iters,
            tol=tol,
            patience=patience,
            min_iters=min_iters,
        )
    elif model_name == "dirgnn_sage":
        e_hat_raw = sage_multiclass_probs(
            labels=state_obs,
            x=x,
            adjacency=adjacency,
            num_layers=num_layers,
            output_dim=output_dim,
            seed=seed,
            use_gpu=use_gpu,
            lr=lr,
            max_iters=max_iters,
            tol=tol,
            patience=patience,
            min_iters=min_iters,
        )
        mu_hat = sage_outcome_surface(
            y=y,
            x=x,
            adjacency=adjacency,
            exposure_obs=t_obs,
            states=EXPOSURE_STATES,
            num_layers=num_layers,
            output_dim=output_dim,
            seed=seed + 1,
            use_gpu=use_gpu,
            lr=lr,
            max_iters=max_iters,
            tol=tol,
            patience=patience,
            min_iters=min_iters,
        )
    else:
        raise ValueError("Unknown model_name.")

    e_hat = _clip_and_renormalize_multiclass_probs(e_hat_raw, clip=float(clip))
    return e_hat, mu_hat


def run(args: argparse.Namespace) -> Path:
    models = ["dirgnn_pna", "dirgnn_sage"]
    z_crit = float(NormalDist().inv_cdf(1.0 - float(args.ci_alpha) / 2.0))
    use_gpu = bool(int(args.use_gpu))
    bw_arg = None if str(args.bandwidth).lower() == "auto" else int(args.bandwidth)

    print(
        f"compare models={models} runs={args.num_runs} n={args.n} clip={args.clip} "
        f"output_dim={args.output_dim} bw={args.bandwidth} variance_method={args.variance_method} "
        f"use_gpu={int(use_gpu)} max_iters={args.max_iters} tol={args.tol} "
        f"patience={args.patience} min_iters={args.min_iters}",
        flush=True,
    )

    stats: dict[str, dict[str, list[float]]] = {}
    for model in models:
        stats[model] = {}
        for est in ESTIMANDS:
            stats[model][f"mse_{est}"] = []
            stats[model][f"cover_{est}"] = []
            stats[model][f"se_{est}"] = []
        stats[model]["bw"] = []

    base_idx = int(STATE_TO_INDEX[BASELINE_STATE])
    started = time.time()
    for run_idx in range(int(args.num_runs)):
        seed = int(args.seed_start) + run_idx
        draw = sample_data_spillover(
            sample_size=int(args.n),
            seed=seed,
            graph_model="rgg",
            p_bidirected=float(args.p_bidirected),
            tau_dir=float(args.tau_dir_true),
            tau_in=float(args.tau_in_true),
            tau_out=float(args.tau_out_true),
        )
        y = np.asarray(draw["Y"], dtype=float).reshape(-1)
        x = np.asarray(draw["node_features"], dtype=float)
        adjacency = np.asarray(draw["adjacency"])
        t_obs = np.asarray(draw["T"], dtype=float)
        state_obs = np.asarray(draw["state_index"], dtype=int).reshape(-1)
        true_taus = draw["true_taus"]
        n_float = float(y.size)

        bw_used, kernels = build_directed_kernels(adjacency, bandwidth=bw_arg)
        for model in models:
            e_hat, mu_hat = evaluate_model(
                model_name=model,
                y=y,
                x=x,
                adjacency=adjacency,
                t_obs=t_obs,
                state_obs=state_obs,
                clip=float(args.clip),
                num_layers=int(args.num_layers),
                output_dim=int(args.output_dim),
                seed=seed,
                use_gpu=use_gpu,
                lr=float(args.lr),
                max_iters=int(args.max_iters),
                tol=float(args.tol),
                patience=int(args.patience),
                min_iters=int(args.min_iters),
            )
            stats[model]["bw"].append(float(bw_used))

            for estimand, target_state in ESTIMAND_TARGETS.items():
                target_idx = int(STATE_TO_INDEX[target_state])
                contrast = _contrast_scores(
                    y=y,
                    observed_state_idx=state_obs,
                    mu_hat=mu_hat,
                    e_hat=e_hat,
                    target_idx=target_idx,
                    baseline_idx=base_idx,
                )
                tau_hat = float(contrast["tau_hat"])
                tau_true = float(true_taus[estimand])
                tau_tilde = np.asarray(contrast["tau_tilde"], dtype=float)
                sigma2 = sigma2_directed(
                    tau_tilde=tau_tilde,
                    kernels=kernels,
                    m_n=n_float,
                    method=str(args.variance_method),
                )
                se = float(math.sqrt(max(sigma2, 0.0) / n_float))
                hw = z_crit * se
                covered = int((tau_hat - hw) <= tau_true <= (tau_hat + hw))

                stats[model][f"mse_{estimand}"].append((tau_hat - tau_true) ** 2)
                stats[model][f"cover_{estimand}"].append(float(covered))
                stats[model][f"se_{estimand}"].append(se)

        if (run_idx + 1) % int(args.log_every) == 0 or (run_idx + 1) == int(args.num_runs):
            elapsed = time.time() - started
            print(f"completed_runs={run_idx + 1}/{args.num_runs} elapsed_sec={elapsed:.1f}", flush=True)

    rows: list[dict] = []
    for model in models:
        row: dict[str, float | int | str] = {
            "model": model,
            "n": int(args.n),
            "num_runs": int(args.num_runs),
            "clip": float(args.clip),
            "output_dim": int(args.output_dim),
            "num_layers": int(args.num_layers),
            "bandwidth": str(args.bandwidth),
            "mean_bandwidth_used": float(np.mean(np.asarray(stats[model]["bw"], dtype=float))),
            "variance_method": str(args.variance_method),
            "ci_alpha": float(args.ci_alpha),
            "z_crit": float(z_crit),
        }

        mse_all = []
        cover_all = []
        ci_gap_all = []
        for estimand in ESTIMANDS:
            mse_e = float(np.mean(np.asarray(stats[model][f"mse_{estimand}"], dtype=float)))
            cover_e = float(np.mean(np.asarray(stats[model][f"cover_{estimand}"], dtype=float)))
            se_e = float(np.mean(np.asarray(stats[model][f"se_{estimand}"], dtype=float)))
            ci_gap_e = abs(cover_e - (1.0 - float(args.ci_alpha)))

            row[f"mse_{estimand}"] = mse_e
            row[f"cover_rate_{estimand}"] = cover_e
            row[f"mean_se_{estimand}"] = se_e
            row[f"ci_gap_{estimand}"] = ci_gap_e

            mse_all.append(mse_e)
            cover_all.append(cover_e)
            ci_gap_all.append(ci_gap_e)

        row["mean_mse_all"] = float(np.mean(np.asarray(mse_all, dtype=float)))
        row["mean_cover_all"] = float(np.mean(np.asarray(cover_all, dtype=float)))
        row["mean_ci_gap_all"] = float(np.mean(np.asarray(ci_gap_all, dtype=float)))
        rows.append(row)

    rows.sort(key=lambda r: (float(r["mean_ci_gap_all"]), float(r["mean_mse_all"])))
    for rank, row in enumerate(rows, start=1):
        row["rank_ci_then_mse"] = int(rank)

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    write_csv(rows, output_csv)

    print("final ranking:", flush=True)
    for row in rows:
        print(
            f"rank={row['rank_ci_then_mse']} model={row['model']} "
            f"cover_all={row['mean_cover_all']:.3f} gap_all={row['mean_ci_gap_all']:.3f} "
            f"tau_tot_cover={row['cover_rate_tau_tot']:.3f} mse_all={row['mean_mse_all']:.6f}",
            flush=True,
        )
    return output_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CI performance between directed dual-channel PNA and directed dual-channel GraphSAGE "
            "under the same local training loop."
        )
    )
    parser.add_argument("--num_runs", type=int, default=30)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed_start", type=int, default=123)
    parser.add_argument("--clip", type=float, default=0.001)
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--bandwidth", type=str, default="auto")
    parser.add_argument("--variance_method", type=str, default="dir_avg")
    parser.add_argument("--ci_alpha", type=float, default=0.05)

    parser.add_argument("--tau_dir_true", type=float, default=-2.0)
    parser.add_argument("--tau_in_true", type=float, default=1.0)
    parser.add_argument("--tau_out_true", type=float, default=1.0)
    parser.add_argument("--p_bidirected", type=float, default=0.05)

    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max_iters", type=int, default=1200)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min_iters", type=int, default=40)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--output_csv", type=str, default="paratune/dirgnn_vs_sage_ci.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    start = time.time()
    output_csv = run(args)
    elapsed = time.time() - start
    print(f"wrote: {output_csv} ({elapsed:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
