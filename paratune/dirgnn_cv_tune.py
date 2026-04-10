#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import PNAConv
from torch_geometric.utils import degree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.GNN import _to_directed_edge_indices
from src.gen_data import sample_data_spillover


@dataclass(frozen=True)
class TuneConfig:
    config_id: str
    output_dim: int
    dropout: float
    weight_decay: float
    width_mode: str
    activation: str


class TunableDirGNN(torch.nn.Module):
    """Directed dual-channel GNN with tunable width split and dropout."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        output_dim: int,
        target_dim: int,
        dropout: float,
        width_mode: str,
        activation: str,
        seed: int,
        deg_hist_in: torch.Tensor,
        deg_hist_out: torch.Tensor,
        aggs: tuple[str, ...] = ("mean", "sum", "std", "min", "max"),
        scalers: tuple[str, ...] = ("identity", "amplification", "attenuation"),
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if int(output_dim) < 2:
            raise ValueError("output_dim must be at least 2.")
        if width_mode not in {"split", "full"}:
            raise ValueError("width_mode must be one of: split, full.")
        if activation not in {"relu", "elu", "gelu", "leaky_relu"}:
            raise ValueError("activation must be one of: relu, elu, gelu, leaky_relu.")

        torch.manual_seed(int(seed))
        self.target_dim = int(target_dim)
        self.num_layers = int(num_layers)
        self.width_mode = str(width_mode)
        self.activation = _activation_module(activation)
        self.dropout = torch.nn.Dropout(float(dropout))

        if self.width_mode == "split":
            in_width = int(output_dim) // 2
            out_width = int(output_dim) - in_width
        else:
            in_width = int(output_dim)
            out_width = int(output_dim)

        self.supply_layers = torch.nn.ModuleList()
        self.demand_layers = torch.nn.ModuleList()
        self.combine_layers = torch.nn.ModuleList()

        prev_width = int(dim)
        for _ in range(self.num_layers):
            self.supply_layers.append(PNAConv(prev_width, in_width, aggs, scalers, deg=deg_hist_in))
            self.demand_layers.append(PNAConv(prev_width, out_width, aggs, scalers, deg=deg_hist_out))
            self.combine_layers.append(torch.nn.Linear(prev_width + in_width + out_width, int(output_dim)))
            prev_width = int(output_dim)

        self.output_layer = torch.nn.Linear(int(output_dim), self.target_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        for layer_idx in range(self.num_layers):
            h_in = self.activation(self.supply_layers[layer_idx](x, data.edge_index_in))
            h_out = self.activation(self.demand_layers[layer_idx](x, data.edge_index_out))
            h_in = self.dropout(h_in)
            h_out = self.dropout(h_out)
            x = self.activation(self.combine_layers[layer_idx](torch.cat([x, h_in, h_out], dim=-1)))
            x = self.dropout(x)

        out = self.output_layer(x)
        if self.target_dim == 1:
            return torch.squeeze(out)
        return out


def parse_float_list(raw: str) -> list[float]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return [float(item) for item in items]


def parse_int_list(raw: str) -> list[int]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated int list.")
    return [int(item) for item in items]


def parse_str_list(raw: str) -> list[str]:
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated string list.")
    return items


def _activation_module(name: str) -> torch.nn.Module:
    key = str(name).lower()
    if key == "relu":
        return torch.nn.ReLU()
    if key == "elu":
        return torch.nn.ELU()
    if key == "gelu":
        return torch.nn.GELU()
    if key == "leaky_relu":
        return torch.nn.LeakyReLU(negative_slope=0.01)
    raise ValueError("Unsupported activation.")


def build_configs(
    output_dims: list[int],
    dropouts: list[float],
    weight_decays: list[float],
    width_modes: list[str],
    activations: list[str],
) -> list[TuneConfig]:
    configs: list[TuneConfig] = []
    for idx, (output_dim, dropout, weight_decay, width_mode, activation) in enumerate(
        itertools.product(output_dims, dropouts, weight_decays, width_modes, activations),
        start=1,
    ):
        configs.append(
            TuneConfig(
                config_id=f"cfg_{idx:03d}",
                output_dim=int(output_dim),
                dropout=float(dropout),
                weight_decay=float(weight_decay),
                width_mode=str(width_mode),
                activation=str(activation),
            )
        )
    return configs


def build_degree_hist(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
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
) -> tuple[int, float]:
    prev_loss: float | None = None
    last_loss = math.nan

    for step in range(int(max_iters)):
        model.train()
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        last_loss = float(loss.detach().cpu().item())
        if prev_loss is not None and abs(prev_loss - last_loss) < float(tol):
            return step + 1, last_loss
        prev_loss = last_loss

    return int(max_iters), last_loss


def make_stratified_folds(labels: np.ndarray, num_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    y = np.asarray(labels, dtype=int).reshape(-1)
    n = int(y.size)
    folds: list[list[int]] = [[] for _ in range(int(num_folds))]

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        parts = np.array_split(idx, int(num_folds))
        for fold_id, part in enumerate(parts):
            if part.size > 0:
                folds[fold_id].extend(part.astype(int).tolist())

    if any(len(fold) == 0 for fold in folds):
        perm = rng.permutation(n)
        parts = np.array_split(perm, int(num_folds))
        return [np.asarray(part, dtype=int) for part in parts]

    return [np.asarray(sorted(fold), dtype=int) for fold in folds]


def train_eval_gps_fold(
    x: np.ndarray,
    state_index: np.ndarray,
    edge_index_in_np: np.ndarray,
    edge_index_out_np: np.ndarray,
    train_mask_np: np.ndarray,
    val_mask_np: np.ndarray,
    cfg: TuneConfig,
    num_layers: int,
    lr: float,
    max_iters: int,
    tol: float,
    seed: int,
    device: torch.device,
) -> float:
    n = int(state_index.size)
    edge_index_in = torch.tensor(edge_index_in_np, dtype=torch.long)
    edge_index_out = torch.tensor(edge_index_out_np, dtype=torch.long)

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index_in=edge_index_in,
        edge_index_out=edge_index_out,
        y=torch.tensor(state_index, dtype=torch.long),
    )

    deg_in = build_degree_hist(data.edge_index_in, num_nodes=n)
    deg_out = build_degree_hist(data.edge_index_out, num_nodes=n)

    torch.manual_seed(int(seed))
    model = TunableDirGNN(
        dim=int(data.num_node_features),
        num_layers=int(num_layers),
        output_dim=int(cfg.output_dim),
        target_dim=8,
        dropout=float(cfg.dropout),
        width_mode=str(cfg.width_mode),
        activation=str(cfg.activation),
        seed=int(seed),
        deg_hist_in=deg_in,
        deg_hist_out=deg_out,
    )

    data = data.to(device)
    model = model.to(device)
    train_mask = torch.as_tensor(train_mask_np, dtype=torch.bool, device=device)
    val_mask = torch.as_tensor(val_mask_np, dtype=torch.bool, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(cfg.weight_decay))
    criterion = torch.nn.CrossEntropyLoss()
    fit_model(data, model, criterion, optimizer, train_mask=train_mask, max_iters=max_iters, tol=tol)

    model.eval()
    with torch.no_grad():
        logits = model(data)
        val_logits = logits[val_mask]
        val_y = data.y[val_mask]
        ce = float(F.cross_entropy(val_logits, val_y).item())
    return ce


def train_eval_outcome_fold(
    x: np.ndarray,
    y: np.ndarray,
    exposure_obs: np.ndarray,
    edge_index_in_np: np.ndarray,
    edge_index_out_np: np.ndarray,
    train_mask_np: np.ndarray,
    val_mask_np: np.ndarray,
    cfg: TuneConfig,
    num_layers: int,
    lr: float,
    max_iters: int,
    tol: float,
    seed: int,
    device: torch.device,
) -> float:
    n = int(y.size)
    x_train = np.concatenate([x, exposure_obs], axis=1)

    edge_index_in = torch.tensor(edge_index_in_np, dtype=torch.long)
    edge_index_out = torch.tensor(edge_index_out_np, dtype=torch.long)

    data = Data(
        x=torch.tensor(x_train, dtype=torch.float),
        edge_index_in=edge_index_in,
        edge_index_out=edge_index_out,
        y=torch.tensor(y, dtype=torch.float),
    )

    deg_in = build_degree_hist(data.edge_index_in, num_nodes=n)
    deg_out = build_degree_hist(data.edge_index_out, num_nodes=n)

    torch.manual_seed(int(seed))
    model = TunableDirGNN(
        dim=int(data.num_node_features),
        num_layers=int(num_layers),
        output_dim=int(cfg.output_dim),
        target_dim=1,
        dropout=float(cfg.dropout),
        width_mode=str(cfg.width_mode),
        activation=str(cfg.activation),
        seed=int(seed),
        deg_hist_in=deg_in,
        deg_hist_out=deg_out,
    )

    data = data.to(device)
    model = model.to(device)
    train_mask = torch.as_tensor(train_mask_np, dtype=torch.bool, device=device)
    val_mask = torch.as_tensor(val_mask_np, dtype=torch.bool, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(cfg.weight_decay))
    criterion = torch.nn.MSELoss()
    fit_model(data, model, criterion, optimizer, train_mask=train_mask, max_iters=max_iters, tol=tol)

    model.eval()
    with torch.no_grad():
        pred = model(data)
        err = pred[val_mask] - data.y[val_mask]
        rmse = float(torch.sqrt(torch.mean(err * err)).item())
    return rmse


def zscore(values: np.ndarray) -> np.ndarray:
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if sigma <= 1e-12:
        return np.zeros_like(values)
    return (values - mu) / sigma


def write_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    output_dims = parse_int_list(args.output_dims)
    dropouts = parse_float_list(args.dropouts)
    weight_decays = parse_float_list(args.weight_decays)
    width_modes = parse_str_list(args.width_modes)
    activations = parse_str_list(args.activations)
    for mode in width_modes:
        if mode not in {"split", "full"}:
            raise ValueError("width_modes must be a comma-separated subset of: split,full")
    for activation in activations:
        if activation not in {"relu", "elu", "gelu", "leaky_relu"}:
            raise ValueError("activations must be a comma-separated subset of: relu,elu,gelu,leaky_relu")

    configs = build_configs(output_dims, dropouts, weight_decays, width_modes, activations)
    data_seeds = [int(args.seed_start) + i for i in range(int(args.num_data_seeds))]

    use_cuda = bool(int(args.use_gpu)) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(
        f"running {len(configs)} configs | folds={args.num_folds} | data_seeds={data_seeds} | device={device.type}",
        flush=True,
    )

    summary_rows: list[dict] = []
    total = len(configs)

    for cfg_idx, cfg in enumerate(configs, start=1):
        fold_rmses: list[float] = []
        fold_ces: list[float] = []

        cfg_start = time.time()
        for data_seed in data_seeds:
            draw = sample_data_spillover(
                sample_size=int(args.n),
                seed=int(data_seed),
                graph_model=str(args.graph_model),
                p_bidirected=float(args.p_bidirected),
                tau_dir=float(args.tau_dir_true),
                tau_in=float(args.tau_in_true),
                tau_out=float(args.tau_out_true),
            )

            x = np.asarray(draw["node_features"], dtype=float)
            y = np.asarray(draw["Y"], dtype=float).reshape(-1)
            t_obs = np.asarray(draw["T"], dtype=float)
            state_index = np.asarray(draw["state_index"], dtype=int).reshape(-1)
            adjacency = np.asarray(draw["adjacency"])
            n = int(y.size)

            edge_index_in_np, edge_index_out_np = _to_directed_edge_indices(adjacency, n=n)
            folds = make_stratified_folds(state_index, num_folds=int(args.num_folds), seed=int(data_seed) + 1009)

            for fold_id, val_idx in enumerate(folds):
                train_mask_np = np.ones(n, dtype=bool)
                train_mask_np[val_idx] = False
                val_mask_np = np.logical_not(train_mask_np)
                model_seed = int(args.model_seed_start) + 10000 * cfg_idx + 101 * data_seed + fold_id

                ce = train_eval_gps_fold(
                    x=x,
                    state_index=state_index,
                    edge_index_in_np=edge_index_in_np,
                    edge_index_out_np=edge_index_out_np,
                    train_mask_np=train_mask_np,
                    val_mask_np=val_mask_np,
                    cfg=cfg,
                    num_layers=int(args.num_layers),
                    lr=float(args.lr),
                    max_iters=int(args.max_iters),
                    tol=float(args.tol),
                    seed=model_seed,
                    device=device,
                )
                rmse = train_eval_outcome_fold(
                    x=x,
                    y=y,
                    exposure_obs=t_obs,
                    edge_index_in_np=edge_index_in_np,
                    edge_index_out_np=edge_index_out_np,
                    train_mask_np=train_mask_np,
                    val_mask_np=val_mask_np,
                    cfg=cfg,
                    num_layers=int(args.num_layers),
                    lr=float(args.lr),
                    max_iters=int(args.max_iters),
                    tol=float(args.tol),
                    seed=model_seed + 1,
                    device=device,
                )

                fold_ces.append(float(ce))
                fold_rmses.append(float(rmse))

        rmse_arr = np.asarray(fold_rmses, dtype=float)
        ce_arr = np.asarray(fold_ces, dtype=float)
        elapsed_cfg = time.time() - cfg_start

        summary_rows.append(
            {
                "config_id": cfg.config_id,
                "output_dim": int(cfg.output_dim),
                "width_mode": str(cfg.width_mode),
                "dropout": float(cfg.dropout),
                "weight_decay": float(cfg.weight_decay),
                "activation": str(cfg.activation),
                "num_layers": int(args.num_layers),
                "num_folds": int(args.num_folds),
                "num_data_seeds": int(args.num_data_seeds),
                "mean_outcome_rmse": float(np.mean(rmse_arr)),
                "std_outcome_rmse": float(np.std(rmse_arr, ddof=1)) if rmse_arr.size > 1 else 0.0,
                "mean_gps_ce": float(np.mean(ce_arr)),
                "std_gps_ce": float(np.std(ce_arr, ddof=1)) if ce_arr.size > 1 else 0.0,
                "n_eval_folds": int(rmse_arr.size),
                "runtime_sec": float(elapsed_cfg),
            }
        )

        print(
            f"[{cfg_idx:02d}/{total}] {cfg.config_id} done "
            f"rmse={float(np.mean(rmse_arr)):.5f} ce={float(np.mean(ce_arr)):.5f} "
            f"time={elapsed_cfg:.1f}s",
            flush=True,
        )

    rmse_means = np.asarray([row["mean_outcome_rmse"] for row in summary_rows], dtype=float)
    ce_means = np.asarray([row["mean_gps_ce"] for row in summary_rows], dtype=float)

    z_rmse = zscore(rmse_means)
    z_ce = zscore(ce_means)

    for idx, row in enumerate(summary_rows):
        row["z_outcome_rmse"] = float(z_rmse[idx])
        row["z_gps_ce"] = float(z_ce[idx])
        row["joint_score"] = 0.5 * float(z_rmse[idx]) + 0.5 * float(z_ce[idx])

    summary_rows.sort(key=lambda item: float(item["joint_score"]))
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = int(rank)

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    write_csv(summary_rows, output_csv)

    top_k = min(int(args.print_top), len(summary_rows))
    print("top configs:", flush=True)
    for row in summary_rows[:top_k]:
        print(
            f"rank={row['rank']:02d} {row['config_id']} "
            f"score={row['joint_score']:.4f} rmse={row['mean_outcome_rmse']:.5f} "
            f"ce={row['mean_gps_ce']:.5f} "
            f"od={row['output_dim']} wm={row['width_mode']} act={row['activation']} "
            f"do={row['dropout']} wd={row['weight_decay']}",
            flush=True,
        )

    return output_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-validated DirGNN hyperparameter tuning for outcome and GPS prediction.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--graph_model", type=str, default="rgg", choices=("rgg", "er"))
    parser.add_argument("--p_bidirected", type=float, default=0.05)
    parser.add_argument("--tau_dir_true", type=float, default=2.0)
    parser.add_argument("--tau_in_true", type=float, default=1.0)
    parser.add_argument("--tau_out_true", type=float, default=1.0)

    parser.add_argument("--num_data_seeds", type=int, default=2)
    parser.add_argument("--seed_start", type=int, default=123)
    parser.add_argument("--num_folds", type=int, default=3)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--output_dims", type=str, default="6,8,12")
    parser.add_argument("--dropouts", type=str, default="0.0,0.1")
    parser.add_argument("--weight_decays", type=str, default="0.0,1e-4")
    parser.add_argument("--width_modes", type=str, default="split,full")
    parser.add_argument("--activations", type=str, default="relu")

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max_iters", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--model_seed_start", type=int, default=2026)
    parser.add_argument("--use_gpu", type=int, default=0)

    parser.add_argument("--print_top", type=int, default=5)
    parser.add_argument("--output_csv", type=str, default="paratune/dirgnn_cv_summary.csv")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    started = time.time()
    output_csv = run(args)
    elapsed = time.time() - started
    print(f"wrote summary table: {output_csv} ({elapsed:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
