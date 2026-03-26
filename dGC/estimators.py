import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import PNAConv
from torch_geometric.utils import degree


def _graph_to_edge_index(graph_or_adjacency, num_nodes: int) -> np.ndarray:
    if hasattr(graph_or_adjacency, "edges"):
        edges = np.asarray(list(graph_or_adjacency.edges()), dtype=np.int64)
        if edges.size == 0:
            return np.empty((2, 0), dtype=np.int64)
    else:
        adjacency = np.asarray(graph_or_adjacency)
        undirected = np.logical_or(adjacency != 0, adjacency.T != 0)
        edges = np.argwhere(np.triu(undirected, 1)).astype(np.int64)
        if edges.size == 0:
            return np.empty((2, 0), dtype=np.int64)
        if adjacency.shape[0] != num_nodes:
            raise ValueError("Adjacency and Y/X row count do not match.")

    edges_rev = edges[:, [1, 0]]
    return np.vstack([edges, edges_rev]).T


def make_gnn_inputs(sample_or_y, X=None, A=None, feature_key: str = "node_features"):
    if isinstance(sample_or_y, dict):
        y_raw = sample_or_y["Y"]
        x_raw = sample_or_y.get(feature_key, sample_or_y["X"])
        a_raw = sample_or_y["adjacency"]
    else:
        y_raw = sample_or_y
        x_raw = X
        a_raw = A

    y = np.asarray(y_raw)
    y = np.squeeze(y).astype(float)
    x = np.asarray(x_raw, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    edge_index = _graph_to_edge_index(a_raw, num_nodes=y.shape[0])
    return y, x, edge_index


class GNN(torch.nn.Module):
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
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_layers = torch.nn.ModuleList()

        prev_width = dim
        for _ in range(num_layers):
            self.hidden_layers.append(
                PNAConv(prev_width, output_dim, aggs, scalers, deg=deg_hist)
            )
            prev_width = output_dim

        self.output_layer = torch.nn.Linear(output_dim, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        x = data.x
        for layer in self.hidden_layers:
            x = self.relu(layer(x, data.edge_index))
        return torch.squeeze(self.output_layer(x))


def train(data, model, criterion, optimizer, sample_mask):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[sample_mask], data.y[sample_mask])
    loss.backward()
    optimizer.step()
    return loss


def GNN_reg(
    Y,
    X=None,
    A=None,
    num_layers=2,
    output_dim=6,
    sample=False,
    seed=0,
    feature_key: str = "node_features",
):
    y, x, edge_index = make_gnn_inputs(Y, X=X, A=A, feature_key=feature_key)

    if isinstance(Y, dict):
        y_for_type = np.asarray(Y["Y"])
    else:
        y_for_type = np.asarray(Y)
    binary_output = np.issubdtype(y_for_type.dtype, np.integer)

    if isinstance(sample, np.ndarray):
        sample_mask = torch.tensor(sample, dtype=torch.bool)
    else:
        sample_mask = torch.ones(y.shape[0], dtype=torch.bool)

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.float),
    )
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=int(d.max()) + 1 if d.numel() > 0 else 1)
    model = GNN(data.num_node_features, num_layers, output_dim, seed, deg_hist=deg)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss() if binary_output else torch.nn.MSELoss()

    old_loss = train(data, model, criterion, optimizer, sample_mask)
    gain = 10.0
    iters = 1
    while gain > 1e-4:
        iters += 1
        loss = train(data, model, criterion, optimizer, sample_mask)
        gain = abs(float(old_loss.item()) - float(loss.item()))
        old_loss = loss
        if iters >= 10000:
            break

    out_final = torch.sigmoid(model(data)) if binary_output else model(data)
    return out_final.detach().numpy()
