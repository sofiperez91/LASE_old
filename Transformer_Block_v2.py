import torch
import torch_geometric as pyg
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class Transformer_Block(nn.Module):
    def __init__(self, c_in, c_out, alpha=0.2, device='cpu'):
        super().__init__()
        self.device = device
        self.W_2 = nn.Parameter(torch.Tensor(1, c_out))
        self.W_3 = nn.Parameter(torch.Tensor(1, c_out))

    def forward(self, x, edge_index, batch_size = 1, use_softmax=False, return_attn_matrix=False):

        ## Linear pass
        x2 = x@torch.diag(self.W_2[0])
        x3 = x@torch.diag(self.W_3[0])
        x4 = x
        num_nodes = x.shape[1]

        ## Get indexes with edges
        x3 = torch.index_select(x3, 1, edge_index[0])
        x4 = torch.index_select(x4, 1, edge_index[1])
     
        ## Calculate attention weights
        attn_weights = torch.einsum("bhc,bhc->bh", x3, x4)

        if use_softmax:
            adj_matrix = torch.Tensor(batch_size, num_nodes, num_nodes)
            for b in range(batch_size):
                adj_matrix[b] = pyg.utils.to_dense_adj(edge_index)
                attn_matrix = attn_weights.new_zeros(adj_matrix.shape).fill_(-9e15)
                attn_matrix[adj_matrix==1] = attn_weights * (2**(-0.5))
                attn_matrix = F.softmax(attn_matrix, dim=1)
        else:
            attn_matrix = torch.Tensor(batch_size, num_nodes, num_nodes) 
            #.to(self.device)
            for b in range(batch_size):
                attn_matrix [b] = pyg.utils.to_dense_adj(edge_index, edge_attr=attn_weights[b].reshape(-1)).to('cpu')
                attn_matrix.to('cpu')
                
                #print(attn_matrix)
            
        ## Calculate final output
        x = torch.einsum("bij,bih->bjh", attn_matrix.to('cpu'), x2.to('cpu'))
        
        if return_attn_matrix:
            return x, attn_matrix
        else:
            return x