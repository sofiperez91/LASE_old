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


MODEL_FILE='../saved_models/glase_unshared_d3_normalized_full_positive_subgraphs_095_unbalanced_complete.pt'
TRAIN_DATA_FILE = './data/sbm3_unbalanced_positive_train_subgraphs_095.pkl'
VAL_DATA_FILE = './data/sbm3_unbalanced_positive_val_subgraphs_095.pkl'
ORIGINAL_GRAPH = './data/sbm3_unbalanced_positive_original_graph_095.pkl'

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

d = 3
gd_steps = 300
device = 'cuda'

model1 = gLASE(d,d, gd_steps)
model1.to(device)

model2 = gLASE(d,d, gd_steps)
model2.to(device)

model3 = gLASE(d,d, gd_steps)
model3.to(device)

q1 = torch.Tensor([[1,1,1]])
q2 = torch.Tensor([[1,1,-1]])
q3 = torch.Tensor([[1,-1,-1]])
Q1=torch.diag(q1[0]).to(device)
Q2=torch.diag(q2[0]).to(device)
Q3=torch.diag(q3[0]).to(device)


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

model3.gd.lin1.weight.data = 4*0.0001*torch.eye(d).to(device)
model3.gd.lin2.weight.data = 4*0.0001*torch.eye(d).to(device)
model3.gd.lin1.weight.requires_grad = False
model3.gd.lin2.weight.requires_grad = False
model3.gd.Q.data = q3.to(device)
model3.gd.Q.requires_grad = False

count_model_1 = 0
count_model_2 = 0
count_model_3 = 0

for i in range(20):
    sub_edge_index, _, _ = dropout_node(data.edge_index, p=0.85)
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
    x_glase3 = model3(x, sub_edge_index, sub_edge_index_2)
    loss1= torch.norm((x_glase1@Q1@x_glase1.T - to_dense_adj(sub_edge_index).squeeze(0)))
    loss2= torch.norm((x_glase2@Q2@x_glase2.T - to_dense_adj(sub_edge_index).squeeze(0)))
    loss3= torch.norm((x_glase3@Q3@x_glase3.T - to_dense_adj(sub_edge_index).squeeze(0)))
    
    arr_loss = np.array([loss1.detach().to('cpu').numpy(), loss2.detach().to('cpu').numpy(), loss3.detach().to('cpu').numpy()])
    print(arr_loss)
    if np.argmin(arr_loss) == 0:
        count_model_1+=1
    elif np.argmin(arr_loss) == 1:
        count_model_2+=1
    else:
        count_model_3+=1
        
arr_count = np.array([count_model_1, count_model_2, count_model_3])
print(arr_count)
        
if np.argmax(arr_count) == 0:
    Q = Q1
elif np.argmax(arr_count) == 1:
    Q = Q2
else:
    Q = Q3
    
    

## PHASE 2

# def get_x_init(num_nodes, alpha, beta):
#     angles = torch.linspace(alpha, beta, num_nodes)
#     x = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
#     return x

def get_x_init(num_nodes, alpha, beta, phi_min, phi_max):
    theta = torch.linspace(alpha, beta, num_nodes)  # polar angle
    phi = torch.linspace(phi_min, phi_max, num_nodes)  # azimuthal angle
    r = 1  # assuming a unit sphere, but you could make this an argument if you want

    x = r * torch.sin(theta).unsqueeze(1) * torch.cos(phi).unsqueeze(1)
    y = r * torch.sin(theta).unsqueeze(1) * torch.sin(phi).unsqueeze(1)
    z = r * torch.cos(theta).unsqueeze(1)

    coords = torch.cat((x, y, z), dim=1)  # concatenate along dimension 1 to get [num_nodes, 3] tensor

    return coords

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
        x = get_x_init(batch.num_nodes, 0, math.pi/2, 0, math.pi/2).to(device)
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
        x = get_x_init(batch.num_nodes, 0, math.pi/2, 0, math.pi/2).to(device)
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