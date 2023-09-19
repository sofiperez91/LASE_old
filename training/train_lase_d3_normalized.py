import sys
sys.path.append("../")

import torch
import numpy as np
import torch_geometric as pyg
from torch_geometric.utils import stochastic_blockmodel_graph, to_dense_adj
from models.RDPG_GD_Unroll_shared_normalized import GD_Unroll
from models.early_stopper import EarlyStopper

num_nodes = 100
n_components = 3
d = 3
device = 'cpu'
epochs = 100000
lr=1e-3
gd_steps = 3
MODEL_FILE='../saved_models/lase_shared_d3_normalized_layers3.pt'

print(gd_steps)
n = [33, 33, 34]

p = [
     [0.9, 0.1, 0.1],
     [0.1, 0.5, 0.1],
     [0.1, 0.1, 0.7]
]

edge_index = stochastic_blockmodel_graph(n, p).to(device)
edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous()
# x = torch.rand(num_nodes,d).to(device)
x = torch.ones(num_nodes,d).to(device)

model = GD_Unroll(d,d, gd_steps)
model.to(device)


# Initialize weights GCN
model.gd.gcn.lins[0].weight.data = torch.eye(d).to(device)
model.gd.gcn.lins[1].weight.data = torch.nn.init.xavier_uniform_(model.gd.gcn.lins[1].weight).to(device)

# Initialize weights TransformerConv
model.gd.gat.lin2.weight.data = torch.nn.init.xavier_uniform_(model.gd.gat.lin2.weight).to(device)
model.gd.gat.lin3.weight.data = torch.eye(d).to(device)
model.gd.gat.lin4.weight.data = torch.eye(d).to(device)

# Freeze weights
model.gd.gcn.lins[0].weight.requires_grad = False
model.gd.gat.lin3.weight.requires_grad = False
model.gd.gat.lin4.weight.requires_grad = False

for param in model.parameters():
    print(param)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=1000, min_delta=0)

min_loss = np.inf

for epoch in range(epochs):
    # Train
    edge_index = stochastic_blockmodel_graph(n, p).to(device)
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, edge_index_2)
    loss = torch.norm((out@out.T - to_dense_adj(edge_index).squeeze(0)))
    
    loss.backward() 
    optimizer.step() 

    if epoch % 500 == 0:
        print(epoch)
        print(f'Train Loss: {loss}')
        torch.save(model.state_dict(), MODEL_FILE)
    
    # if early_stopper.early_stop(loss):      
    #     break

torch.save(model.state_dict(), MODEL_FILE)

