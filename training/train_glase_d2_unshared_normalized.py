import sys
sys.path.append("../")

import torch
import numpy as np
import torch_geometric as pyg
from torch_geometric.utils import stochastic_blockmodel_graph, to_dense_adj, erdos_renyi_graph
from models.GRDPG_GD_Unroll_unshared_normalized import gLASE
from models.bigbird_attention import big_bird_attention
from models.early_stopper import EarlyStopper
from graspologic.embed import AdjacencySpectralEmbed 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
from tqdm import tqdm

num_nodes = 100
d = 3
device = 'cuda'
epochs = 50000
lr=1e-3
gd_steps = 5
MODEL_FILE='../saved_models/glase_unshared_d3_normalized_full.pt'
TRAIN_DATA_FILE='./data/sbm3_neg_train.pkl'
VAL_DATA_FILE='./data/sbm3_neg_val.pkl'

n = [50, 50]
p = [
     [0.1, 0.7],
     [0.7, 0.1]
]
edge_index = stochastic_blockmodel_graph(n, p).to(device)
edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)

x = torch.rand((num_nodes, d)).to(device)

model = gLASE(d,d, gd_steps)
model.to(device)

for step in range(gd_steps):
    model.gd[step].lin1.weight.data = lr*torch.eye(d).to(device)#torch.nn.init.xavier_uniform_(model.gd[step].lin1.weight)*lr
    model.gd[step].lin2.weight.data = lr*torch.eye(d).to(device)#torch.nn.init.xavier_uniform_(model.gd[step].lin2.weight)*lr
    model.gd[step].Q.data = (torch.rand([1,d])-0.5).to(device)#torch.ones([1,d]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=10, min_delta=0)

with open(TRAIN_DATA_FILE, 'rb') as f:
    df_train = pickle.load(f)
df_train = [Data(x = data.x, edge_index = data.edge_index) for data in df_train]

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)
df_val = [Data(x = data.x, edge_index = data.edge_index) for data in df_val]

train_loader=  DataLoader(df_train, batch_size=1, shuffle = True)
val_loader =  DataLoader(df_val, batch_size=1, shuffle = False)

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
        out = model(batch.x, batch.edge_index, edge_index_2)
        I_pq = torch.diag(model.gd[0].Q[0])
        loss = torch.norm((out@I_pq@out.T - to_dense_adj(edge_index).squeeze(0)))
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
        edge_index_2 = erdos_renyi_graph(num_nodes, 0.5, directed=False).to(device) 
        out = model(batch.x, batch.edge_index, edge_index_2)
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

    if early_stopper.early_stop(val_loss_epoch[epoch]):    
        optimal_epoch = np.argmin(val_loss_epoch)
        print("Optimal epoch: ", optimal_epoch)         
        break
