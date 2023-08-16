import torch
import torch_geometric as pyg
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class Transformer_Block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.W_2 = nn.Parameter(torch.Tensor(1, c_out))
        self.W_3 = nn.Parameter(torch.Tensor(1, c_out))

    def forward(self, x, edge_index, use_softmax=False, return_attn_matrix=False):

        ## Linear pass
        x2 = x@torch.diag(self.W_2[0])
        x3 = x@torch.diag(self.W_3[0])
        x4 = x

        ## Get indexes with edges
        x3 = torch.index_select(x3, 0, edge_index[0])
        x4 = torch.index_select(x4, 0, edge_index[1])
     
        ## Calculate attention weights
        attn_weights = torch.einsum("hc,hc->h", x3, x4)
        attn_weights = attn_weights.reshape(-1)
        
        adj_matrix = to_dense_adj(edge_index)

        if use_softmax:
            attn_matrix = attn_weights.new_zeros(adj_matrix.shape).fill_(-9e15)
            attn_matrix[adj_matrix==1] = attn_weights * (2**(-0.5))
            attn_matrix = F.softmax(attn_matrix, dim=1).squeeze(0)
        else:
            attn_matrix = to_dense_adj(edge_index, edge_attr=attn_weights).squeeze(0)

                
                #print(attn_matrix)
            
        ## Calculate final output
        x = torch.einsum("ij,ih->jh", attn_matrix, x2)
        
        if return_attn_matrix:
            return x, attn_matrix
        else:
            return x