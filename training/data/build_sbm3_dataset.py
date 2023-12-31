import sys
sys.path.append("../")

import torch
import pickle
from torch_geometric.utils import stochastic_blockmodel_graph
from torch_geometric.data import Data

num_nodes = 150
d = 3
device = 'cpu'
TRAIN_DATA_FILE='./sbm3_unbalanced_train_2.pkl'
VAL_DATA_FILE='./sbm3_unbalanced_val_2.pkl'

n = [70, 50, 30]

p = [
     [0.5, 0.1, 0.3],
     [0.1, 0.9, 0.2], 
     [0.3, 0.2, 0.7]
]

df_train = []
df_val = []

for j in range(2000):
    x = torch.rand((num_nodes, d))
    edge_index = stochastic_blockmodel_graph(n, p).to(device)
    data = Data(x = x, edge_index = edge_index)
    if j < 1600:
        df_train.append(data)
    else:
        df_val.append(data)

with open(TRAIN_DATA_FILE, 'wb') as f:
    pickle.dump(df_train, f)
with open(VAL_DATA_FILE, 'wb') as f:
    pickle.dump(df_val, f)