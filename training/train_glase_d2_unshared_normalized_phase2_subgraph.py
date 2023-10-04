import sys
sys.path.append("../")

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, stochastic_blockmodel_graph
from models.GRDPG_GD_Unroll_shared_normalized import gLASE
from models.GRDPG_GD_Unroll_unshared_normalized import gLASE as gLASE_w
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pickle
import numpy as np
from models.early_stopper import EarlyStopper
import time
from torch.optim.lr_scheduler import StepLR
from models.rsvd import rsvd
import math

def get_x_init(num_nodes, alpha, beta):
    angles = torch.linspace(alpha, beta, num_nodes)
    x = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

    return x

num_nodes = 100
d = 2
device = 'cuda'

MODEL_FILE='../saved_models/glase_unshared_d2_normalized_full_negative_phase2_subgraphs_095_unbalanced_init_pos.pt'
TRAIN_DATA_FILE = './data/sbm2_train_subgraphs_negative_095_unbalanced.pkl'
VAL_DATA_FILE = './data/sbm2_val_subgraphs_negative_095_unbalanced.pkl'

#### LOAD DATA ####

with open(TRAIN_DATA_FILE, 'rb') as f:
    df_train = pickle.load(f)

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)

train_loader=  DataLoader(df_train, batch_size=1, shuffle = True)
val_loader =  DataLoader(df_val, batch_size=1, shuffle = False)   

#### PHASE 2 ####
lr=1e-3
gd_steps = 5
epochs = 10000

edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
mask = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)

Q = torch.Tensor([[1,0],[0,1]]).to(device)
print('Starting PHASE 2 training')
print('Q=',Q)

model2 = gLASE_w(d,d, gd_steps)
model2.to(device)    

## Initialization
for step in range(gd_steps):
    model2.gd[step].lin1.weight.data = torch.nn.init.xavier_uniform_(model2.gd[step].lin1.weight)*lr
    model2.gd[step].lin2.weight.data = torch.nn.init.xavier_uniform_(model2.gd[step].lin2.weight)*lr

optimizer = torch.optim.Adam(model2.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=10, min_delta=0)

train_loss_epoch=[]
val_loss_epoch=[]
min_val_loss = np.inf

start = time.time()

for epoch in range(epochs):
    # Train
    train_loss_step=[]
    model2.train()
    train_loop = tqdm(train_loader)
    for i, batch in enumerate(train_loop):  
        batch.to(device) 
        optimizer.zero_grad()
        x = get_x_init(batch.num_nodes, 0, math.pi/2).to(device)
        out = model2(x, batch.edge_index, batch.edge_index_2, Q, batch.edge_index_2)
        loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index).squeeze(0)))
        loss.backward() 
        optimizer.step() 
        
        train_loss_step.append(loss.detach().to('cpu').numpy())
        train_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        train_loop.set_postfix(loss=loss)

    train_loss_epoch.append(np.array(train_loss_step).mean())
        
    # Validation
    val_loss_step=[] 
    model2.eval()
    val_loop = tqdm(val_loader)
    for i, batch in enumerate(val_loop):
        batch.to(device)      
        x = get_x_init(batch.num_nodes, 0, math.pi/2).to(device)
        out = model2(x, batch.edge_index, batch.edge_index_2, Q, batch.edge_index_2)
        loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index).squeeze(0)))

        val_loss_step.append(loss.detach().to('cpu').numpy())
        val_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        val_loop.set_postfix(loss=loss)

    val_loss_epoch.append(np.array(val_loss_step).mean())

    if val_loss_epoch[epoch] < min_val_loss:
        torch.save(model2.state_dict(), MODEL_FILE)
        min_val_loss = val_loss_epoch[epoch]
        print("Best model updated")
        print("Val loss: ", min_val_loss)

    if early_stopper.early_stop(val_loss_epoch[epoch]):    
        optimal_epoch = np.argmin(val_loss_epoch)
        print("Optimal epoch: ", optimal_epoch)         
        break    

stop = time.time()
print(f"Training time: {stop - start}s")