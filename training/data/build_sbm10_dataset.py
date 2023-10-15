import sys
sys.path.append("../")

import torch
import pickle
from torch_geometric.utils import stochastic_blockmodel_graph
from torch_geometric.data import Data

num_nodes = 650
d = 8
device = 'cpu'
TRAIN_DATA_FILE='./data/sbm8_neg_train.pkl'
VAL_DATA_FILE='./data/sbm8_neg_val.pkl'


n = [100, 50, 150, 50, 50, 100, 50, 100]

p = [
    [0.1, 0.9, 0.7, 0.9, 0.8, 0.6, 0.8, 0.7],
    [0.9, 0.6, 0.8, 0.9, 0.7, 0.9, 0.7, 0.9],
    [0.7, 0.8, 0.3, 0.8, 0.7, 0.7, 0.9, 0.8],
    [0.9, 0.9, 0.8, 0.4, 0.8, 0.6, 0.8, 0.7],
    [0.8, 0.7, 0.7, 0.8, 0.1, 0.8, 0.7, 0.6],
    [0.6, 0.9, 0.7, 0.6, 0.8, 0.6, 0.8, 0.7],
    [0.8, 0.7, 0.9, 0.8, 0.7, 0.8, 0.7, 0.9],
    [0.7, 0.9, 0.8, 0.7, 0.6, 0.7, 0.9, 0.4]
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
    if j % 100 == 0:
        print(j)

with open(TRAIN_DATA_FILE, 'wb') as f:
    pickle.dump(df_train, f)
with open(VAL_DATA_FILE, 'wb') as f:
    pickle.dump(df_val, f)