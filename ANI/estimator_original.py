import torch, numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import PNAConv

class GNN(torch.nn.Module):
    """PyTorch class implementing PNAConv GNN architecture.

    Parameters
    ----------
    dim : int
        Dimension of node features.
    num_layers : int
        Number of hidden layers.
    output_dim : int
        Dimension of each GNN hidden layer.
    seed : int
        Seed for initializing parameters.
    """
    def __init__(self, dim, num_layers, output_dim, seed=0, deg_hist=None, aggs=['mean','sum','std','min','max'], scalers=['identity','amplification','attenuation']): 
        if num_layers < 1:
            raise ValueError('Must have at least one hidden layer.')
        super().__init__()
        torch.manual_seed(seed) # for parameter initialization
        self.num_layers = num_layers

        self.hidden_layers = torch.nn.ModuleList()
        prev_width = dim
        for l in range(self.num_layers):
            self.hidden_layers.append(PNAConv(prev_width, output_dim, aggs, scalers, deg=deg_hist))
            if l==0: prev_width=output_dim

        self.output_layer = torch.nn.Linear(output_dim, 1)
        self.ReLU = torch.nn.ReLU()

    # forward pass
    def forward(self, data):
        x = data.x
        for l in range(self.num_layers):
            x = self.ReLU(self.hidden_layers[l](x, data.edge_index))
        return torch.squeeze(self.output_layer(x)) 

def train(data, model, criterion, optimizer, sample):
    """Generic training function for a PyTorch neural network.

    Parameters
    ----------
    data : PyTorch data object
        Contains at least two parameters: x (node feature matrix), y (node scalar outcomes).
    model : PyTorch neural network object
        Neural network architecture, e.g. the GNN class above.
    criterion : function
        Loss function.
    optimizer : function
        PyTorch optimizer.
    sample : numpy array
        Vector of booleans for inclusion of each unit in the sample. This is used to only utilize a subsample for training.

    Returns
    -------
    loss : scalar
        Value of loss function.
    out : PyTorch tensor
        Output of final layer.
    """
    optimizer.zero_grad() 
    out = model(data)
    loss = criterion(out[sample], data.y[sample])
    loss.backward()
    optimizer.step()
    return loss, out

def GNN_reg(Y, X, A, num_layers, output_dim, sample=False, seed=0):
    """Nonparametric regression using GNN.

    Parameters
    ----------
    Y : numpy array
        n x 1 vector of outcomes. NOTE: If outcomes are binary, this must be formatted as an array of ints.
    X : numpy array
        n x d node feature matrix.
    A : NetworkX graph
        Undirected network on n nodes formatted as a NetworkX object.
    num_layers : int
        Number of GNN layers.
    output_dim : int
        Dimension of the output of each GNN hidden layer.
    sample : numpy array
        Vector of booleans for inclusion of each unit in the sample during training. Set to False to use entire sample for training.
    seed : int
        Seed for random number generation.

    Returns
    -------
    out_final : numpy array
        Vector of predictions, one for each node.
    """
    if type(sample) != np.ndarray: 
        sample = np.ones(Y.size, dtype=bool) # handle case where sample=False
    binary_output = True if np.issubdtype(Y.dtype, np.integer) else False # detect if Y is binary

    # set up data as PyTorch Geometric data object
    edgelist = np.array(A.edges)
    edgelist = np.vstack([ edgelist, edgelist[:,[1,0]] ]).T # make undirected
    data = Data(
        x=torch.tensor(X, dtype=torch.float), 
        edge_index=torch.tensor(edgelist, dtype=torch.long),
        y=torch.tensor(Y, dtype=torch.float)
    )
    # node degree histogram tensor used by PNAConv scalers
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long) # degree list
    deg = torch.bincount(d, minlength=int(d.max()) + 1) # degree histogram (counts)
    model = GNN(data.num_node_features, num_layers, output_dim, seed, deg_hist=deg)

    # gradient descent
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss() if binary_output else torch.nn.MSELoss()
    old_loss, out = train(data, model, criterion, optimizer, sample)
    gain = 10
    iters = 1
    while gain > 1e-4:
        iters += 1
        loss, out = train(data, model, criterion, optimizer, sample)
        gain = abs(old_loss - loss)
        old_loss = loss
        if iters >= 10000: 
            print('{estimator} max iters reached.')
            break

    out_final = torch.sigmoid(model(data)) if binary_output else model(data)
    return out_final.detach().numpy()
