import sys
sys.path.append("../")

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from models.GRDPG_GD_Unroll_unshared_normalized import gLASE as gLASE_w
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pickle
import numpy as np
from models.early_stopper import EarlyStopper
import time
import math


num_nodes = 650
d = 8
device = 'cuda'

MODEL_FILE='../saved_models/glase_unshared_d8_normalized_full_negative_phase2.pt'
TRAIN_DATA_FILE='./data/sbm8_neg_train.pkl'
VAL_DATA_FILE='./data/sbm8_neg_val.pkl'


import torch

def get_x_init(num_nodes, alpha, beta, phi_min, phi_max):
    dim = 8  # target dimensionality
    r = 1  # assuming a unit hypersphere
    
    # Generate num_nodes angles for each dimension, within specified ranges
    angles = [torch.linspace(phi_min, phi_max, num_nodes) for _ in range(dim - 1)]
    angles.insert(0, torch.linspace(alpha, beta, num_nodes))  # Insert theta angles at the beginning
    
    coords = torch.zeros((num_nodes, dim))
    
    # Compute coordinates in a loop
    for i in range(dim):
        coord = r
        for j in range(i):
            coord *= torch.sin(angles[j])
        if i < dim - 1:
            coord *= torch.cos(angles[i])
        coords[:, i] = coord
        
    return coords


#### LOAD DATA ####

with open(TRAIN_DATA_FILE, 'rb') as f:
    df_train = pickle.load(f)
df_train = [Data(x = data.x, edge_index = data.edge_index) for data in df_train]

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)
df_val = [Data(x = data.x, edge_index = data.edge_index) for data in df_val]


train_loader=  DataLoader(df_train, batch_size=1, shuffle = True)
val_loader =  DataLoader(df_val, batch_size=1, shuffle = False)

start = time.time()

#### PHASE 2 ####
lr=1e-4
gd_steps = 5
epochs = 10000

edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
mask = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)

q = torch.Tensor([[1,1,-1,-1,-1,-1,-1,-1]])
Q=torch.diag(q[0]).to(device)
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

for epoch in range(epochs):
    # Train
    train_loss_step=[]
    model2.train()
    train_loop = tqdm(train_loader)
    for i, batch in enumerate(train_loop):  
        batch.to(device) 
        optimizer.zero_grad()   
        x  = get_x_init(num_nodes,  0, math.pi/2, 0, math.pi/2).to(device)
        out = model2(x, batch.edge_index, edge_index_2, Q, mask)
        loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))
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
        x  = get_x_init(num_nodes,  0, math.pi/2, 0, math.pi/2).to(device)     
        
        out = model2(x, batch.edge_index, edge_index_2, Q, mask)
        loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))

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