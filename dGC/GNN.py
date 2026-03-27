import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import PNAConv
from torch_geometric.utils import degree


# ==============================
# Graph -> Edge Index Utilities
# ==============================
def _to_edge_index(A, n: int) -> np.ndarray:
    adjacency = np.asarray(A)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be a square matrix.")
    if adjacency.shape[0] != n:
        raise ValueError("Adjacency and Y/X row count do not match.")
    undirected = np.logical_or(adjacency != 0, adjacency.T != 0)
    edges = np.argwhere(np.triu(undirected, 1)).astype(np.int64)
    if edges.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    return np.vstack([edges, edges[:, [1, 0]]]).T


def _to_directed_edge_indices(A, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return edge indices for in-channel and out-channel aggregations.

    in-channel: aggregate from upstream neighbors j->i (uses original direction).
    out-channel: aggregate from downstream neighbors i->k (reverse edges so k->i messages).
    """
    adjacency = np.asarray(A)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be a square matrix.")
    if adjacency.shape[0] != n:
        raise ValueError("Adjacency and Y/X row count do not match.")

    directed = adjacency != 0
    np.fill_diagonal(directed, False)
    edges_in = np.argwhere(directed).astype(np.int64)
    if edges_in.size == 0:
        empty = np.empty((2, 0), dtype=np.int64)
        return empty, empty

    edge_index_in = edges_in.T
    edge_index_out = edge_index_in[[1, 0], :]
    return edge_index_in, edge_index_out


# ====================
# Undirected GNN Model
# ====================
class GNN(torch.nn.Module):
    """PyTorch class implementing PNAConv GNN architecture."""

    def __init__(
        self,
        dim,
        num_layers,
        output_dim,
        seed=0,
        deg_hist=None,
        aggs=("mean", "sum", "std", "min", "max"),
        scalers=("identity", "amplification", "attenuation"),
    ):
        if num_layers < 1:
            raise ValueError("Must have at least one hidden layer.")
        super().__init__()
        torch.manual_seed(seed)
        self.num_layers = num_layers

        self.hidden_layers = torch.nn.ModuleList()
        prev_width = dim
        for l in range(self.num_layers):
            self.hidden_layers.append(PNAConv(prev_width, output_dim, aggs, scalers, deg=deg_hist))
            if l == 0:
                prev_width = output_dim

        self.output_layer = torch.nn.Linear(output_dim, 1)
        self.ReLU = torch.nn.ReLU()

    def forward(self, data):
        x = data.x
        for l in range(self.num_layers):
            x = self.ReLU(self.hidden_layers[l](x, data.edge_index))
        return torch.squeeze(self.output_layer(x))


# ==========================
# Directed Dual-Channel GNN
# ==========================
class DirGNN(torch.nn.Module):
    """Dual-channel directed GNN with separate in/out aggregations per layer."""

    def __init__(
        self,
        dim,
        num_layers,
        output_dim,
        seed=0,
        deg_hist_in=None,
        deg_hist_out=None,
        aggs=("mean", "sum", "std", "min", "max"),
        scalers=("identity", "amplification", "attenuation"),
    ):
        if num_layers < 1:
            raise ValueError("Must have at least one hidden layer.")
        super().__init__()
        torch.manual_seed(seed)
        self.num_layers = num_layers
        self.relu = torch.nn.ReLU()

        self.supply_layers = torch.nn.ModuleList()
        self.demand_layers = torch.nn.ModuleList()
        self.combine_layers = torch.nn.ModuleList()

        prev_width = dim
        for _ in range(self.num_layers):
            self.supply_layers.append(PNAConv(prev_width, output_dim, aggs, scalers, deg=deg_hist_in))
            self.demand_layers.append(PNAConv(prev_width, output_dim, aggs, scalers, deg=deg_hist_out))
            self.combine_layers.append(torch.nn.Linear(prev_width + 2 * output_dim, output_dim))
            prev_width = output_dim

        self.output_layer = torch.nn.Linear(output_dim, 1)

    def forward(self, data):
        x = data.x
        for l in range(self.num_layers):
            h_in = self.relu(self.supply_layers[l](x, data.edge_index_in))
            h_out = self.relu(self.demand_layers[l](x, data.edge_index_out))
            x = self.relu(self.combine_layers[l](torch.cat([x, h_in, h_out], dim=-1)))
        return torch.squeeze(self.output_layer(x))


# ==================
# Training Utilities
# ==================
def train(data, model, criterion, optimizer, sample):
    """Generic training function for a PyTorch neural network."""
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[sample], data.y[sample])
    loss.backward()
    optimizer.step()
    return loss, out


# ====================
# Regression Entrypoints
# ====================
def GNN_reg(
    Y,
    X,
    A,
    num_layers=2,
    output_dim=6,
    sample=False,
    seed=0,
):
    """Nonparametric regression using GNN."""
    Y = np.asarray(Y)
    Y = np.squeeze(Y)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = int(Y.shape[0])

    if not isinstance(sample, np.ndarray):
        sample = np.ones(n, dtype=bool)
    binary_output = np.issubdtype(Y.dtype, np.integer)

    edgelist = _to_edge_index(A, n=n)
    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(edgelist, dtype=torch.long),
        y=torch.tensor(Y, dtype=torch.float),
    )

    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=int(d.max()) + 1 if d.numel() > 0 else 1)
    model = GNN(data.num_node_features, num_layers, output_dim, seed, deg_hist=deg)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss() if binary_output else torch.nn.MSELoss()
    old_loss, _ = train(data, model, criterion, optimizer, sample)
    gain = 10.0
    iters = 1
    while gain > 1e-4:
        iters += 1
        loss, _ = train(data, model, criterion, optimizer, sample)
        gain = abs(float(old_loss.item()) - float(loss.item()))
        old_loss = loss
        if iters >= 10000:
            print("GNN_reg max iters reached.")
            break

    out_final = torch.sigmoid(model(data)) if binary_output else model(data)
    return out_final.detach().numpy()


def GNN_reg_dir(
    Y,
    X,
    A,
    num_layers=2,
    output_dim=6,
    sample=False,
    seed=0,
):
    """Nonparametric regression using a directed dual-channel GNN."""
    Y = np.asarray(Y)
    Y = np.squeeze(Y)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = int(Y.shape[0])

    if not isinstance(sample, np.ndarray):
        sample = np.ones(n, dtype=bool)
    binary_output = np.issubdtype(Y.dtype, np.integer)

    edge_index_in, edge_index_out = _to_directed_edge_indices(A, n=n)
    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index_in=torch.tensor(edge_index_in, dtype=torch.long),
        edge_index_out=torch.tensor(edge_index_out, dtype=torch.long),
        y=torch.tensor(Y, dtype=torch.float),
    )

    d_in = degree(data.edge_index_in[1], num_nodes=data.num_nodes, dtype=torch.long)
    d_out = degree(data.edge_index_out[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg_in = torch.bincount(d_in, minlength=int(d_in.max()) + 1 if d_in.numel() > 0 else 1)
    deg_out = torch.bincount(d_out, minlength=int(d_out.max()) + 1 if d_out.numel() > 0 else 1)
    model = DirGNN(
        data.num_node_features,
        num_layers,
        output_dim,
        seed,
        deg_hist_in=deg_in,
        deg_hist_out=deg_out,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss() if binary_output else torch.nn.MSELoss()
    old_loss, _ = train(data, model, criterion, optimizer, sample)
    gain = 10.0
    iters = 1
    while gain > 1e-4:
        iters += 1
        loss, _ = train(data, model, criterion, optimizer, sample)
        gain = abs(float(old_loss.item()) - float(loss.item()))
        old_loss = loss
        if iters >= 10000:
            print("GNN_reg_dir max iters reached.")
            break

    out_final = torch.sigmoid(model(data)) if binary_output else model(data)
    return out_final.detach().numpy()


# ==========
# Smoke Test
# ==========
if __name__ == "__main__":
    try:
        from .gen_data import sample_data_simple
    except ImportError:
        try:
            from dGC.gen_data import sample_data_simple
        except ImportError:
            from gen_data import sample_data_simple

    draw = sample_data_simple(sample_size=200, seed=123, graph_model="rgg", tau=2.0, p_treat=0.5)

    x = np.asarray(draw["node_features"], dtype=float)
    a = draw["adjacency"]
    y_true = np.asarray(draw["Y"], dtype=float)
    d_true = np.asarray(draw["D"], dtype=int)

    y_fit = np.asarray(GNN_reg(Y=y_true, X=x, A=a, num_layers=2, output_dim=6, seed=123), dtype=float)
    p_fit = np.asarray(
        GNN_reg(Y=d_true.astype(int), X=x, A=a, num_layers=2, output_dim=6, seed=456),
        dtype=float,
    )

    print("Outcome fit (first 10):")
    for i in range(10):
        print(f"i={i:02d} y_fit={y_fit[i]:.6f} y_true={y_true[i]:.6f}")

    print("Propensity fit (first 10):")
    for i in range(10):
        print(f"i={i:02d} p_fit={p_fit[i]:.6f} d_true={int(d_true[i])}")
