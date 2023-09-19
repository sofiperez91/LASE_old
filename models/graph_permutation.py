import torch
from torch_geometric.utils import to_dense_adj

def graph_permutation(edge_index, indexes):
    A = to_dense_adj(edge_index).squeeze(0).numpy()
    A_new = A.copy()

    for i, index in enumerate(indexes):
        ## Rows
        old_row = A_new[i].copy()
        new_row = A_new[index].copy()
        A_new[i] = new_row
        A_new[index] = old_row

        ## Columns    
        old_col = A_new[:,i].copy()
        new_col = A_new[:,index].copy()
        A_new[:,i] = new_col
        A_new[:,index] = old_col

    return torch.tensor(A_new).nonzero().t().contiguous()