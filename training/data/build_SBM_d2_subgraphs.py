import sys
sys.path.append("../")

import torch
from torch_geometric.utils import stochastic_blockmodel_graph, to_dense_adj, erdos_renyi_graph
from torch_geometric.data import Data
import pickle
from torch_geometric.utils import dropout_node

# BUILD ORIGINAL GRAPH
d = 2
num_nodes = 12000
n = [9000, 3000]
# p = [
#      [0.9, 0.1],
#      [0.1, 0.5]
# ]

# p = [
#      [0.3, 0.9],
#      [0.9, 0.3]
# ]

# p = [
#      [0.3, 0.8],
#      [0.8, 0.6]
# 

p = [
     [0.9, 0.2],
     [0.2, 0.6]
]


edge_index = stochastic_blockmodel_graph(n, p)

with open('./data/sbm2_original_graph_positive_095_unbalanced.pkl', 'wb') as f:
    pickle.dump(Data(edge_index=edge_index, num_nodes=num_nodes), f)

# CREATE SUBGRAPHS
train_data = []
val_data = []
for i in range(1000):
    print(i)
    sub_edge_index, _, _ = dropout_node(edge_index, p=0.95)
    adj_matrix = to_dense_adj(sub_edge_index).squeeze(0)
    non_zero_rows = (adj_matrix.sum(dim=1) != 0)
    adj_matrix = adj_matrix[non_zero_rows]
    adj_matrix = adj_matrix[:, non_zero_rows]
    sub_edge_index = adj_matrix.nonzero().t().contiguous()  
    n_nodes = adj_matrix.shape[0]
    # x = torch.ones(n_nodes,d)  
    x = torch.rand((n_nodes, d))
    edge_index_2 = torch.ones([n_nodes,n_nodes],).nonzero().t().contiguous()
    ER07 = erdos_renyi_graph(n_nodes, 0.7, directed=False)
    ER05 = erdos_renyi_graph(n_nodes, 0.5, directed=False)
    ER03 = erdos_renyi_graph(n_nodes, 0.3, directed=False)
    if i < 800:
        train_data.append(Data(x=x, edge_index=sub_edge_index, edge_index_2=edge_index_2, ER07=ER07, ER05=ER05, ER03=ER03, num_nodes=n_nodes))
    else:
        val_data.append(Data(x=x, edge_index=sub_edge_index, edge_index_2=edge_index_2, ER07=ER07, ER05=ER05, ER03=ER03, num_nodes=n_nodes))
        
with open('./data/sbm2_train_subgraphs_positive_095_unbalanced.pkl', 'wb') as f:
    pickle.dump(train_data, f)
    
with open('./data/sbm2_val_subgraphs_positive_095_unbalanced.pkl', 'wb') as f:
    pickle.dump(val_data, f)