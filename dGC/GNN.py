import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import PNAConv
from torch_geometric.utils import degree


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


def train(data, model, criterion, optimizer, sample):
    """Generic training function for a PyTorch neural network."""
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[sample], data.y[sample])
    loss.backward()
    optimizer.step()
    return loss, out


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
