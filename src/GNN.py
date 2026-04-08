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
        target_dim=1,
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

        self.target_dim = int(target_dim)
        self.output_layer = torch.nn.Linear(output_dim, self.target_dim)
        self.ReLU = torch.nn.ReLU()

    def forward(self, data):
        x = data.x
        for l in range(self.num_layers):
            x = self.ReLU(self.hidden_layers[l](x, data.edge_index))
        out = self.output_layer(x)
        if self.target_dim == 1:
            return torch.squeeze(out)
        return out


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
        target_dim=1,
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
        self.target_dim = int(target_dim)
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

        self.output_layer = torch.nn.Linear(output_dim, self.target_dim)

    def forward(self, data):
        x = data.x
        for l in range(self.num_layers):
            h_in = self.relu(self.supply_layers[l](x, data.edge_index_in))
            h_out = self.relu(self.demand_layers[l](x, data.edge_index_out))
            x = self.relu(self.combine_layers[l](torch.cat([x, h_in, h_out], dim=-1)))
        out = self.output_layer(x)
        if self.target_dim == 1:
            return torch.squeeze(out)
        return out


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


def _fit_model(data, model, criterion, optimizer, sample_t):
    old_loss, _ = train(data, model, criterion, optimizer, sample_t)
    gain = 10.0
    iters = 1
    while gain > 1e-4:
        iters += 1
        loss, _ = train(data, model, criterion, optimizer, sample_t)
        gain = abs(float(old_loss.item()) - float(loss.item()))
        old_loss = loss
        if iters >= 10000:
            break


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
    use_gpu=True,
):
    """Nonparametric regression using GNN."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    Y = np.asarray(Y)
    Y = np.squeeze(Y)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = int(Y.shape[0])

    if not isinstance(sample, np.ndarray):
        sample = np.ones(n, dtype=bool)
    sample_t = torch.as_tensor(sample, dtype=torch.bool, device=device)
    binary_output = np.issubdtype(Y.dtype, np.integer)

    edgelist = _to_edge_index(A, n=n)
    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(edgelist, dtype=torch.long),
        y=torch.tensor(Y, dtype=torch.float),
    )

    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=int(d.max()) + 1 if d.numel() > 0 else 1)
    model = GNN(data.num_node_features, num_layers, output_dim, target_dim=1, seed=seed, deg_hist=deg)
    data = data.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss() if binary_output else torch.nn.MSELoss()
    _fit_model(data, model, criterion, optimizer, sample_t)

    out_final = torch.sigmoid(model(data)) if binary_output else model(data)
    return out_final.detach().cpu().numpy()


def GNN_reg_dir(
    Y,
    X,
    A,
    num_layers=2,
    output_dim=6,
    sample=False,
    seed=0,
    use_gpu=True,
):
    """Nonparametric regression using a directed dual-channel GNN."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    Y = np.asarray(Y)
    Y = np.squeeze(Y)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = int(Y.shape[0])

    if not isinstance(sample, np.ndarray):
        sample = np.ones(n, dtype=bool)
    sample_t = torch.as_tensor(sample, dtype=torch.bool, device=device)
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
        target_dim=1,
        seed=seed,
        deg_hist_in=deg_in,
        deg_hist_out=deg_out,
    )
    data = data.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss() if binary_output else torch.nn.MSELoss()
    _fit_model(data, model, criterion, optimizer, sample_t)

    out_final = torch.sigmoid(model(data)) if binary_output else model(data)
    return out_final.detach().cpu().numpy()


def GNN_reg_multiclass(
    labels,
    X,
    A,
    num_classes=8,
    num_layers=2,
    output_dim=6,
    sample=False,
    seed=0,
    use_gpu=True,
):
    """Multiclass classification with an undirected GNN."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = int(y.shape[0])
    if X.shape[0] != n:
        raise ValueError("labels and features must have the same length.")
    if not isinstance(sample, np.ndarray):
        sample = np.ones(n, dtype=bool)
    sample_t = torch.as_tensor(sample, dtype=torch.bool, device=device)

    edgelist = _to_edge_index(A, n=n)
    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(edgelist, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=int(d.max()) + 1 if d.numel() > 0 else 1)
    model = GNN(data.num_node_features, num_layers, output_dim, target_dim=int(num_classes), seed=seed, deg_hist=deg)
    data = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    _fit_model(data, model, criterion, optimizer, sample_t)
    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def GNN_reg_dir_multiclass(
    labels,
    X,
    A,
    num_classes=8,
    num_layers=2,
    output_dim=6,
    sample=False,
    seed=0,
    use_gpu=True,
):
    """Multiclass classification with a directed dual-channel GNN."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = int(y.shape[0])
    if X.shape[0] != n:
        raise ValueError("labels and features must have the same length.")
    if not isinstance(sample, np.ndarray):
        sample = np.ones(n, dtype=bool)
    sample_t = torch.as_tensor(sample, dtype=torch.bool, device=device)

    edge_index_in, edge_index_out = _to_directed_edge_indices(A, n=n)
    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index_in=torch.tensor(edge_index_in, dtype=torch.long),
        edge_index_out=torch.tensor(edge_index_out, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )
    d_in = degree(data.edge_index_in[1], num_nodes=data.num_nodes, dtype=torch.long)
    d_out = degree(data.edge_index_out[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg_in = torch.bincount(d_in, minlength=int(d_in.max()) + 1 if d_in.numel() > 0 else 1)
    deg_out = torch.bincount(d_out, minlength=int(d_out.max()) + 1 if d_out.numel() > 0 else 1)
    model = DirGNN(
        data.num_node_features,
        num_layers,
        output_dim,
        target_dim=int(num_classes),
        seed=seed,
        deg_hist_in=deg_in,
        deg_hist_out=deg_out,
    )
    data = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    _fit_model(data, model, criterion, optimizer, sample_t)
    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def _outcome_surface_from_model(model, data_base, X_base, states):
    preds = []
    with torch.no_grad():
        for state in states:
            exposure = np.repeat(np.asarray(state, dtype=float)[None, :], repeats=X_base.shape[0], axis=0)
            x_eval = np.concatenate([X_base, exposure], axis=1)
            data_eval = Data(**{k: v for k, v in data_base.items()})
            data_eval.x = torch.tensor(x_eval, dtype=torch.float, device=data_base["x"].device)
            pred = model(data_eval).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    return np.column_stack(preds).astype(float)


def GNN_reg_outcome_surface(
    Y,
    X,
    A,
    exposure_obs,
    states,
    num_layers=2,
    output_dim=6,
    seed=0,
    use_gpu=True,
):
    """Fit one joint outcome surface and predict for each exposure state."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    y = np.asarray(Y, dtype=float).reshape(-1)
    x = np.asarray(X, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    exp_obs = np.asarray(exposure_obs, dtype=float)
    if exp_obs.ndim != 2 or exp_obs.shape[1] != 3:
        raise ValueError("exposure_obs must be shape (n, 3).")
    if exp_obs.shape[0] != y.size or x.shape[0] != y.size:
        raise ValueError("Y, X, and exposure_obs must have matching first dimension.")

    x_train = np.concatenate([x, exp_obs], axis=1)
    n = int(y.size)
    edgelist = _to_edge_index(A, n=n)
    data = Data(
        x=torch.tensor(x_train, dtype=torch.float),
        edge_index=torch.tensor(edgelist, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.float),
    )

    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=int(d.max()) + 1 if d.numel() > 0 else 1)
    model = GNN(data.num_node_features, num_layers, output_dim, target_dim=1, seed=seed, deg_hist=deg)
    data = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
    sample_t = torch.ones(n, dtype=torch.bool, device=device)
    _fit_model(data, model, criterion, optimizer, sample_t)

    data_base = {
        "x": data.x,
        "edge_index": data.edge_index,
        "y": data.y,
    }
    return _outcome_surface_from_model(model, data_base, x, states)


def GNN_reg_dir_outcome_surface(
    Y,
    X,
    A,
    exposure_obs,
    states,
    num_layers=2,
    output_dim=6,
    seed=0,
    use_gpu=True,
):
    """Fit one directed joint outcome surface and predict for each exposure state."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    y = np.asarray(Y, dtype=float).reshape(-1)
    x = np.asarray(X, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    exp_obs = np.asarray(exposure_obs, dtype=float)
    if exp_obs.ndim != 2 or exp_obs.shape[1] != 3:
        raise ValueError("exposure_obs must be shape (n, 3).")
    if exp_obs.shape[0] != y.size or x.shape[0] != y.size:
        raise ValueError("Y, X, and exposure_obs must have matching first dimension.")

    x_train = np.concatenate([x, exp_obs], axis=1)
    n = int(y.size)
    edge_index_in, edge_index_out = _to_directed_edge_indices(A, n=n)
    data = Data(
        x=torch.tensor(x_train, dtype=torch.float),
        edge_index_in=torch.tensor(edge_index_in, dtype=torch.long),
        edge_index_out=torch.tensor(edge_index_out, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.float),
    )

    d_in = degree(data.edge_index_in[1], num_nodes=data.num_nodes, dtype=torch.long)
    d_out = degree(data.edge_index_out[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg_in = torch.bincount(d_in, minlength=int(d_in.max()) + 1 if d_in.numel() > 0 else 1)
    deg_out = torch.bincount(d_out, minlength=int(d_out.max()) + 1 if d_out.numel() > 0 else 1)
    model = DirGNN(
        data.num_node_features,
        num_layers,
        output_dim,
        target_dim=1,
        seed=seed,
        deg_hist_in=deg_in,
        deg_hist_out=deg_out,
    )
    data = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
    sample_t = torch.ones(n, dtype=torch.bool, device=device)
    _fit_model(data, model, criterion, optimizer, sample_t)

    preds = []
    with torch.no_grad():
        for state in states:
            exposure = np.repeat(np.asarray(state, dtype=float)[None, :], repeats=x.shape[0], axis=0)
            x_eval = np.concatenate([x, exposure], axis=1)
            data_eval = Data(
                x=torch.tensor(x_eval, dtype=torch.float, device=device),
                edge_index_in=data.edge_index_in,
                edge_index_out=data.edge_index_out,
                y=data.y,
            )
            pred = model(data_eval).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    return np.column_stack(preds).astype(float)


# ==========
# Smoke Test
# ==========
if __name__ == "__main__":
    try:
        from .gen_data import sample_data_simple
    except ImportError:
        try:
            from src.gen_data import sample_data_simple
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
