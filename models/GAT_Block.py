import torch
import torch_geometric as pyg
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class GAT_Block(nn.Module):
    def __init__(self, c_in, c_out, alpha=0.2):
        super().__init__()

        self.lin = nn.Linear(c_in, c_out, bias=False)
        self.a = nn.Parameter(Tensor(1, 2*c_out))
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, edge_index, use_softmax=False, return_attn_matrix=False):

        ## Linear pass
        x = self.lin(x)

        ## Concatenate inputs
        a_input = torch.cat(
            [
                torch.index_select(x, 0, edge_index[0]),
                torch.index_select(x, 0, edge_index[1]),
            ],
            dim=-1,
        )

        ## Calculate attention weights
        attn_weights = torch.einsum("hc,hc->h", a_input, self.a)
        attn_weights = self.leakyrelu(attn_weights)
        attn_weights = attn_weights.reshape(-1)

        adj_matrix = to_dense_adj(edge_index)
        
        if use_softmax:
            attn_matrix = attn_weights.new_zeros(adj_matrix.shape).fill_(-9e15)
            attn_matrix[adj_matrix==1] = attn_weights
            attn_matrix = F.softmax(attn_matrix, dim=1).squeeze(0)
        else:
            attn_matrix = to_dense_adj(edge_index, edge_attr=attn_weights).squeeze(0)
            
        ## Calculate final output
        x = torch.einsum("ij,ih->jh", attn_matrix, x)
        
        if return_attn_matrix:
            return x, attn_matrix
        else:
            return x