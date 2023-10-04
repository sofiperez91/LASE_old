import torch
import torch.nn as nn
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_adj

class GD_Block(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.Q = nn.Parameter(torch.Tensor(1, out_channels))
        self.activation = nn.Tanh()

    def forward(self, input, edge_index, edge_index_2): 
        
        # ## Binary weights
        # prob = torch.sigmoid(self.Q)
        # binary_Q = torch.where(torch.rand_like(self.Q) < prob, torch.ones_like(self.Q), -torch.ones_like(self.Q))[0]
        
        cliped_Q = self.activation(self.Q[0])
        
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
        attn_matrix = to_dense_adj(edge_index_2, edge_attr=attn_weights).squeeze(0)

        ## Calculate final output
        x_2 = torch.einsum("ij,ih->jh", attn_matrix, x2)@torch.diag(self.Q[0])

        return x_1 - x_2 #+ self.Q - self.Q.detach()


import torch.nn as nn
from torch_geometric.nn import Sequential

class gLASE(nn.Module):
    def __init__(self, in_channels, out_channels, gd_steps):
        super().__init__()

        self.gd_steps = gd_steps
        self.gd = GD_Block(in_channels,out_channels)
        self.activation = nn.Tanh()

    def forward(self, input, edge_index, edge_index_2): 
        x = input
        for i in range(self.gd_steps):
             x = self.gd(x, edge_index, edge_index_2)
        return x#, self.activation(self.gd.Q[0])
