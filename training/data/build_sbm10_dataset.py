import sys
sys.path.append("../")

import torch
import pickle
from torch_geometric.utils import stochastic_blockmodel_graph
from torch_geometric.data import Data

num_nodes = 1000
d = 10
device = 'cpu'
TRAIN_DATA_FILE='./sbm10_train.pkl'
VAL_DATA_FILE='./sbm10_val.pkl'

n = [int(num_nodes/2), int(num_nodes/2),int(num_nodes/2), int(num_nodes/2),int(num_nodes/2), int(num_nodes/2),int(num_nodes/2), int(num_nodes/2),int(num_nodes/2), int(num_nodes/2)]

p = [
     [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1],
     [0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1],
     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1],
     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1],
     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6],
]

df_train = []
df_val = []

for j in range(1000):
    x = torch.rand((num_nodes, d))
    edge_index = stochastic_blockmodel_graph(n, p).to(device)
    data = Data(x = x, edge_index = edge_index)
    if j < 800:
        df_train.append(data)
    else:
        df_val.append(data)

with open(TRAIN_DATA_FILE, 'wb') as f:
    pickle.dump(df_train, f)
with open(VAL_DATA_FILE, 'wb') as f:
    pickle.dump(df_val, f)