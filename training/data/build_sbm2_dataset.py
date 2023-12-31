import sys
sys.path.append("../")

import torch
import pickle
from torch_geometric.utils import stochastic_blockmodel_graph
from torch_geometric.data import Data

num_nodes = 100
d = 2
device = 'cpu'
TRAIN_DATA_FILE='./sbm2_neg_train_2_large.pkl'
VAL_DATA_FILE='./sbm2_neg_val_2_large.pkl'

n = [50, 50]

p = [
     [0.3, 0.9],
     [0.9, 0.3]
]

df_train = []
df_val = []

for j in range(4000):
    x = torch.rand((num_nodes, d))
    edge_index = stochastic_blockmodel_graph(n, p).to(device)
    data = Data(x = x, edge_index = edge_index)
    if j < 3200:
        df_train.append(data)
    else:
        df_val.append(data)

with open(TRAIN_DATA_FILE, 'wb') as f:
    pickle.dump(df_train, f)
with open(VAL_DATA_FILE, 'wb') as f:
    pickle.dump(df_val, f)