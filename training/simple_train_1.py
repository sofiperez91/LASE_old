'''
Prueba 1
- Shared layers
- No restrictions over weights.
'''


import sys
sys.path.append("../")

import torch
import numpy as np
import torch_geometric as pyg
from torch_geometric.utils import stochastic_blockmodel_graph, to_dense_adj
from models.RDPG_GD_Unroll_shared import GD_Unroll
import matplotlib.pyplot as plt
import seaborn as sns
from models.RDPG_GD import RDPG_GD_Armijo

num_nodes = 100
n_components = 2
d = 2
device = 'cpu'
epochs = 750
lr=1e-4
gd_steps = 5
MODEL_FILE='../saved_models/lase_shared_d2_v0.pt'

print(gd_steps)
n = [int(num_nodes/2), int(num_nodes/2)]

p = [
     [0.9, 0.1],
     [0.1, 0.5]
]

edge_index = stochastic_blockmodel_graph(n, p).to(device)
edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous()
x = torch.rand(num_nodes,d).to(device)

model = GD_Unroll(d,d, gd_steps)
model.to(device)


# Initialize weights GCN
model.gd.gcn.lins[0].weight.data = torch.eye(d).to(device)
model.gd.gcn.lins[1].weight.data = lr*torch.nn.init.xavier_uniform_(model.gd.gcn.lins[1].weight).to(device)

# Initialize weights TransformerConv
model.gd.gat.lin2.weight.data = lr*torch.nn.init.xavier_uniform_(model.gd.gat.lin2.weight).to(device)
model.gd.gat.lin3.weight.data = torch.eye(d).to(device)
model.gd.gat.lin4.weight.data = torch.eye(d).to(device)
# model.gd.gat.lin3.weight.data = lr*torch.nn.init.xavier_uniform_(model.gd.gat.lin3.weight).to(device)
# model.gd.gat.lin4.weight.data = lr*torch.nn.init.xavier_uniform_(model.gd.gat.lin4.weight).to(device)

# Freeze weights
# model.gd.gcn.lins[0].weight.requires_grad = False
# model.gd.gat.lin3.weight.requires_grad = False
# model.gd.gat.lin4.weight.requires_grad = False

for param in model.parameters():
    print(param)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    # To achieve generalization, we must train with multiple realizations.
    edge_index = stochastic_blockmodel_graph(n, p).to(device)
    edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous()
    x = torch.rand(num_nodes,d).to(device)
    # Train
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, edge_index_2)
    loss = torch.norm((out@out.T - to_dense_adj(edge_index).squeeze(0)))
    
    loss.backward() 
    optimizer.step() 

    if epoch % 10 == 0:
        print(f'Train Loss: {loss}')

torch.save(model.state_dict(), MODEL_FILE)

for param in model.parameters():
    print(param)

## Pruebo con otra realizacion

edge_index = stochastic_blockmodel_graph(n, p).to(device)
edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous()
x = torch.rand(num_nodes, d).to(device)

model.eval()
x_lase = model(x, edge_index, edge_index_2)
x_lase = x_lase.detach()
loss = torch.norm((x_lase@x_lase.T - to_dense_adj(edge_index).squeeze(0)))
print(f"Loss LASE shared weights ({gd_steps} layers): ", loss)
x_lase = x_lase.to('cpu')

x_gd, cost_gd, iter = RDPG_GD_Armijo(x, edge_index)
x_gd = x_gd.to('cpu')
loss = torch.norm((x_gd@x_gd.T - to_dense_adj(edge_index).squeeze(0)))
print("Loss RDPG GD Armijo: ", loss)
print("Armijo iterations: ", iter)



## Aumento numero de nodos

num_nodes = 250
n = [int(num_nodes/2), int(num_nodes/2)]

edge_index = stochastic_blockmodel_graph(n, p).to(device)
edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous()

x = torch.rand(num_nodes, d).to(device)
x_lase = model(x, edge_index, edge_index_2)
x_lase = x_lase.detach()

loss = torch.norm((x_lase@x_lase.T - to_dense_adj(edge_index).squeeze(0)))
print(f"Loss LASE shared weights ({gd_steps} layers): ", loss)

x_lase = x_lase.to('cpu')

x_gd, cost_gd, iter = RDPG_GD_Armijo(x, edge_index)
x_gd = x_gd.to('cpu')
loss = torch.norm((x_gd@x_gd.T - to_dense_adj(edge_index).squeeze(0)))
print("Loss RDPG GD Armijo: ", loss)
print("Armijo iterations: ", iter)
