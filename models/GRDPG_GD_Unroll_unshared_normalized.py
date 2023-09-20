import torch.nn as nn
from torch_geometric.nn import TAGConv, Sequential
from torch_geometric.nn.conv import MessagePassing

class GD_Block(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.Q = nn.Parameter(torch.Tensor(1, out_channels))

    def forward(self, input, edge_index, edge_index_2): 
        
        ## GCN Layer
        x_1 = self.propagate(edge_index, x=input)
        x_1 = input + self.lin1.forward(x_1)@torch.diag(self.Q[0])


        ## GAT Layer
        x2 = self.lin2(input)
        _x3 = input@torch.diag(self.Q[0])
        _x4 = input

        ## Get indexes with edges
        x3 = torch.index_select(_x3, 0, edge_index_2[0])
        x4 = torch.index_select(_x4, 0, edge_index_2[1])

        ## Calculate attention weights
        attn_weights = torch.einsum("hc,hc->h", x3, x4)
        attn_weights = attn_weights.reshape(-1)
        adj_matrix = to_dense_adj(edge_index_2)
        attn_matrix = to_dense_adj(edge_index_2, edge_attr=attn_weights).squeeze(0)

        ## Calculate final output
        x_2 = torch.einsum("ij,ih->jh", attn_matrix, x2)@torch.diag(self.Q[0])

        return x_1 - x_2


import torch.nn as nn
from torch_geometric.nn import Sequential

class gLASE(nn.Module):
    def __init__(self, in_channels, out_channels, gd_steps):
        super().__init__()

        self.gd_steps = gd_steps
        self.activation = nn.Tanh()
        layers = []

        for _ in range(gd_steps):
            layers.append((GD_Block(in_channels, out_channels), 'x, edge_index, edge_index_2 -> x'))
        self.gd = Sequential('x, edge_index, edge_index_2', [layer for layer in layers])

    def forward(self, input, edge_index, edge_index_2): 
        x = input
        x = self.gd(x, edge_index, edge_index_2)
                
        return x
