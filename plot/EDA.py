#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

# Allow `python plot/EDA.py` to import from repo root `src/`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gen_data import sample_data_dir


def _build_digraph(adjacency: np.ndarray) -> nx.DiGraph:
    a = np.asarray(adjacency)
    n = int(a.shape[0])
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    edges = np.argwhere(a != 0)
    g.add_edges_from((int(i), int(j)) for i, j in edges)
    return g


def _binary_colors(values: np.ndarray, color_false: str = "#bdbdbd", color_true: str = "#d62728") -> list[str]:
    v = np.asarray(values, dtype=int).reshape(-1)
    return [color_true if int(x) == 1 else color_false for x in v]


def _add_binary_legend(ax, values: np.ndarray, label_true: str) -> None:
    v = np.asarray(values, dtype=int).reshape(-1)
    p_red = float(np.mean(v == 1))
    p_gray = 1.0 - p_red
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#d62728",
            markeredgecolor="none",
            markersize=8,
            label=f"{label_true}: {p_red:.2%}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#bdbdbd",
            markeredgecolor="none",
            markersize=8,
            label=f"Not {label_true.lower()}: {p_gray:.2%}",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)


def _draw_directed_panel(
    ax,
    g: nx.DiGraph,
    pos: dict,
    node_color,
    title: str,
) -> None:
    nx.draw_networkx_edges(
        g,
        pos=pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        width=0.9,
        alpha=0.85,
        edge_color="#6c757d",
        connectionstyle="arc3,rad=0.06",
        min_source_margin=10,
        min_target_margin=12,
    )
    nx.draw_networkx_nodes(
        g,
        pos=pos,
        ax=ax,
        node_size=180,
        node_color=node_color,
        edgecolors="white",
        linewidths=0.6,
    )
    nx.draw_networkx_labels(g, pos=pos, ax=ax, font_size=8)
    ax.set_title(title, fontsize=13, pad=10, weight="semibold")
    ax.set_aspect("equal")
    ax.axis("off")


def _set_shared_bounds(axes, pos: dict[int, np.ndarray], pad: float = 0.12) -> None:
    coords = np.asarray(list(pos.values()), dtype=float)
    xmin, ymin = np.min(coords, axis=0)
    xmax, ymax = np.max(coords, axis=0)
    dx = max(float(xmax - xmin), 1e-6)
    dy = max(float(ymax - ymin), 1e-6)
    xpad = pad * dx
    ypad = pad * dy
    for ax in axes:
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)


def main() -> None:
    draw = sample_data_dir(sample_size=20, seed=123, graph_model="rgg")
    adjacency = np.asarray(draw["adjacency"], dtype=int)
    d = np.asarray(draw["D"], dtype=int)

    upstream_treated = ((adjacency.T @ d) > 0).astype(int)
    downstream_treated = ((adjacency @ d) > 0).astype(int)

    g = _build_digraph(adjacency)
    # Use one stable layout for all panels so differences are only color encoding.
    pos = nx.spring_layout(nx.Graph(g), seed=123, k=1.1, iterations=500)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()

    _draw_directed_panel(ax0, g, pos, "#9ecae1", "Directed Graph (n=20)")
    _draw_directed_panel(ax1, g, pos, _binary_colors(d), "1) Treated (red) vs Untreated (gray)")
    _add_binary_legend(ax1, d, label_true="Treated")

    _draw_directed_panel(
        ax2,
        g,
        pos,
        _binary_colors(upstream_treated),
        "2) Upstream has treated neighbor (red)",
    )
    _add_binary_legend(ax2, upstream_treated, label_true="Upstream treated")

    _draw_directed_panel(
        ax3,
        g,
        pos,
        _binary_colors(downstream_treated),
        "3) Downstream has treated neighbor (red)",
    )
    _add_binary_legend(ax3, downstream_treated, label_true="Downstream treated")
    _set_shared_bounds([ax0, ax1, ax2, ax3], pos)

    fig.suptitle("Directed Network EDA (n=20)", fontsize=18, weight="bold")

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "eda_dir_n50.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved_plot={out_path}")


if __name__ == "__main__":
    main()
