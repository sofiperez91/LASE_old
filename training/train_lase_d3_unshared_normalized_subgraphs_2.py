import sys
sys.path.append("../")

import torch
import numpy as np
import torch_geometric as pyg
from torch_geometric.utils import stochastic_blockmodel_graph, to_dense_adj
from models.RDPG_GD_Unroll_unshared_normalized import GD_Unroll
import matplotlib.pyplot as plt
import seaborn as sns
from models.RDPG_GD import RDPG_GD_Armijo
from models.early_stopper import EarlyStopper
from graspologic.embed import AdjacencySpectralEmbed 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
from tqdm import tqdm
from torch_geometric.loader import ClusterLoader, ClusterData
from torch_geometric.utils import dropout_node


d = 3
device = 'cpu'
epochs = 50000
lr=1e-3
gd_steps = 5
MODEL_FILE = '../saved_models/lase_unshared_d3_normalized_subgraphs_large_095_unbalanced.pt'


# BUILD ORIGINAL GRAPH
num_nodes = 12000
n = [6000, 4000, 2000]
p = [
     [0.9, 0.1, 0.1],
     [0.1, 0.5, 0.1],
     [0.1, 0.1, 0.7]
]
edge_index = stochastic_blockmodel_graph(n, p).to(device)

with open('./data/edge_index_d3_subgraphs_large_095_unbalanced.pkl', 'wb') as f:
    pickle.dump(Data(edge_index=edge_index, num_nodes=num_nodes), f)

# CREATE SUBGRAPHS
train_data = []
val_data = []
for i in range(1000):
    sub_edge_index, _, _ = dropout_node(edge_index, p=0.95)
    adj_matrix = to_dense_adj(sub_edge_index).squeeze(0)
    non_zero_rows = (adj_matrix.sum(dim=1) != 0)
    adj_matrix = adj_matrix[non_zero_rows]
    adj_matrix = adj_matrix[:, non_zero_rows]
    sub_edge_index = adj_matrix.nonzero().t().contiguous()  
    n_nodes = adj_matrix.shape[0]
    x = torch.ones(n_nodes,d)  
    edge_index_2 = torch.ones([n_nodes,n_nodes],).nonzero().t().contiguous()
    if i < 800:
        train_data.append(Data(x=x, edge_index=sub_edge_index, edge_index_2=edge_index_2))
    else:
        val_data.append(Data(x=x, edge_index=sub_edge_index, edge_index_2=edge_index_2))
        
    
train_loader =  DataLoader(train_data, batch_size=1, shuffle = True)
val_loader =  DataLoader(val_data, batch_size=1, shuffle = False)
    
model = GD_Unroll(d,d, gd_steps)
model.to(device)

# Initialization

for step in range(gd_steps):
    # TAGConv
    model.gd[step].gcn.lins[0].weight.data = torch.eye(d).to(device)
    model.gd[step].gcn.lins[0].weight.requires_grad = False
    model.gd[step].gcn.lins[1].weight.data = torch.nn.init.xavier_uniform_(model.gd[step].gcn.lins[1].weight)*lr

    # TransformerBlock
    model.gd[step].gat.lin2.weight.data = lr*torch.nn.init.xavier_uniform_(model.gd[step].gat.lin2.weight.data).to(device)
    model.gd[step].gat.lin3.weight.data = torch.eye(d).to(device)
    model.gd[step].gat.lin3.weight.requires_grad = False
    model.gd[step].gat.lin4.weight.data = torch.eye(d).to(device)
    model.gd[step].gat.lin4.weight.requires_grad = False
    

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=10, min_delta=0)


sample = next(iter(train_loader))

adj_matrix = to_dense_adj(sample.edge_index).squeeze(0).numpy()
ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='full')
Xhats = ase.fit_transform(adj_matrix)
best_error = str(np.linalg.norm(Xhats@Xhats.T-adj_matrix))
print("best error: ",best_error )


train_loss_epoch=[]
val_loss_epoch=[]
min_val_loss = np.inf

for epoch in range(epochs):
    # Train
    train_loss_step=[]
    model.train()
    train_loop = tqdm(train_loader)
    for i, batch in enumerate(train_loop):  
        batch.to(device) 
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_index_2)
        loss = torch.norm((out@out.T - to_dense_adj(batch.edge_index).squeeze(0)))
        loss.backward() 
        optimizer.step() 

        train_loss_step.append(loss.detach().to('cpu').numpy())
        train_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        train_loop.set_postfix(loss=loss)

    train_loss_epoch.append(np.array(train_loss_step).mean())
        
    # Validation
    val_loss_step=[] 
    model.eval()
    val_loop = tqdm(val_loader)
    for i, batch in enumerate(val_loop):
        batch.to(device)      
        out = model(batch.x, batch.edge_index, batch.edge_index_2)
        loss = torch.norm((out@out.T - to_dense_adj(batch.edge_index).squeeze(0)))

        val_loss_step.append(loss.detach().to('cpu').numpy())
        val_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        val_loop.set_postfix(loss=loss)

    val_loss_epoch.append(np.array(val_loss_step).mean())

    if val_loss_epoch[epoch] < min_val_loss:
        torch.save(model.state_dict(), MODEL_FILE)
        min_val_loss = val_loss_epoch[epoch]
        print("Best model updated")
        print("Val loss: ", min_val_loss)
        print("Best error: ",best_error)

    if early_stopper.early_stop(val_loss_epoch[epoch]):    
        optimal_epoch = np.argmin(val_loss_epoch)
        print("Optimal epoch: ", optimal_epoch)         
        break
