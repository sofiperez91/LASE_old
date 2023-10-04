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
from torch_geometric.utils import dropout_node


MODEL_FILE='../saved_models/glase_unshared_d2_normalized_full_positive_subgraphs_095_unbalanced_complete.pt'
TRAIN_DATA_FILE = './data/sbm2_train_subgraphs_positive_095_unbalanced.pkl'
VAL_DATA_FILE = './data/sbm2_val_subgraphs_positive_095_unbalanced.pkl'
ORIGINAL_GRAPH = './data/sbm2_original_graph_positive_095_unbalanced.pkl'

## Load data
with open(TRAIN_DATA_FILE, 'rb') as f:
    df_train = pickle.load(f)

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)


train_loader=  DataLoader(df_train, batch_size=1, shuffle = True)
val_loader =  DataLoader(df_val, batch_size=1, shuffle = False)   



## PHASE 1
with open(ORIGINAL_GRAPH, 'rb') as f:
    data= pickle.load(f)

d = 2
gd_steps = 100
device = 'cuda'

model1 = gLASE(d,d, gd_steps)
model1.to(device)

model2 = gLASE(d,d, gd_steps)
model2.to(device)

q1 = torch.Tensor([[1,1]])
q2 = torch.Tensor([[1,-1]])
Q1=torch.diag(q1[0]).to(device)
Q2=torch.diag(q2[0]).to(device)


### Initialization
model1.gd.lin1.weight.data = 4*0.0001*torch.eye(d).to(device)
model1.gd.lin2.weight.data = 4*0.0001*torch.eye(d).to(device)
model1.gd.lin1.weight.requires_grad = False
model1.gd.lin2.weight.requires_grad = False
model1.gd.Q.data = q1.to(device)
model1.gd.Q.requires_grad = False

model2.gd.lin1.weight.data = 4*0.0001*torch.eye(d).to(device)
model2.gd.lin2.weight.data = 4*0.0001*torch.eye(d).to(device)
model2.gd.lin1.weight.requires_grad = False
model2.gd.lin2.weight.requires_grad = False
model2.gd.Q.data = q2.to(device)
model2.gd.Q.requires_grad = False


count_model_1 = 0
count_model_2 = 0

for i in range(20):
    sub_edge_index, _, _ = dropout_node(data.edge_index, p=0.90)
    adj_matrix = to_dense_adj(sub_edge_index).squeeze(0)
    non_zero_rows = (adj_matrix.sum(dim=1) != 0)
    adj_matrix = adj_matrix[non_zero_rows]
    adj_matrix = adj_matrix[:, non_zero_rows]
    sub_edge_index = adj_matrix.nonzero().t().contiguous().to(device)  
    n_nodes = adj_matrix.shape[0]
    sub_edge_index_2 = torch.ones([n_nodes,n_nodes],).nonzero().t().contiguous().to(device)

    x = torch.rand((n_nodes, d)).to(device) 
    x_glase1 = model1(x, sub_edge_index, sub_edge_index_2)
    x_glase2 = model2(x, sub_edge_index, sub_edge_index_2)
    loss1= torch.norm((x_glase1@Q1@x_glase1.T - to_dense_adj(sub_edge_index).squeeze(0)))
    loss2= torch.norm((x_glase2@Q2@x_glase2.T - to_dense_adj(sub_edge_index).squeeze(0)))

    if loss1 >= loss2:
        count_model_2+=1
    else:
        count_model_1+=1



if count_model_1 > count_model_2:
    Q = Q1
else: 
    Q = Q2


## PHASE 2

def get_x_init(num_nodes, alpha, beta):
    angles = torch.linspace(alpha, beta, num_nodes)
    x = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    return x

lr=1e-3
gd_steps = 5
epochs = 10000

print('Starting PHASE 2 training')
print('Q=',Q)

model = gLASE_w(d,d, gd_steps)
model.to(device)    

## Initialization
for step in range(gd_steps):
    model.gd[step].lin1.weight.data = torch.nn.init.xavier_uniform_(model.gd[step].lin1.weight)*lr
    model.gd[step].lin2.weight.data = torch.nn.init.xavier_uniform_(model.gd[step].lin2.weight)*lr

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=10, min_delta=0)

train_loss_epoch=[]
val_loss_epoch=[]
min_val_loss = np.inf

start = time.time()

for epoch in range(epochs):
    # Train
    train_loss_step=[]
    model.train()
    train_loop = tqdm(train_loader)
    for i, batch in enumerate(train_loop):  
        batch.to(device) 
        optimizer.zero_grad()
        x = get_x_init(batch.num_nodes, 0, math.pi/2).to(device)
        out = model(x, batch.edge_index, batch.edge_index_2, Q, batch.edge_index_2)
        loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index).squeeze(0)))
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
        x = get_x_init(batch.num_nodes, 0, math.pi/2).to(device)
        out = model(x, batch.edge_index, batch.edge_index_2, Q, batch.edge_index_2)
        loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index).squeeze(0)))

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

stop = time.time()
print(f"Training time: {stop - start}s")