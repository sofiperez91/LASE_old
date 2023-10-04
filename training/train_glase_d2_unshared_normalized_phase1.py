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
from models.rsvd import rsvd

num_nodes = 100
d = 2
device = 'cuda'
epochs = 50
lr=1e-2
gd_steps = 100
MODEL_FILE='../saved_models/glase_unshared_d2_normalized_full_negative_rsvd.pt'
TRAIN_DATA_FILE='./data/sbm2_neg_train_2.pkl'
VAL_DATA_FILE='./data/sbm2_neg_val_2.pkl'

#### LOAD DATA ####

with open(TRAIN_DATA_FILE, 'rb') as f:
    df_train = pickle.load(f)
df_train = [Data(x = data.x, edge_index = data.edge_index) for data in df_train]

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)
df_val = [Data(x = data.x, edge_index = data.edge_index) for data in df_val]

train_loader=  DataLoader(df_train, batch_size=1, shuffle = True)
val_loader =  DataLoader(df_val, batch_size=1, shuffle = False)


mask = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)

#### PHASE 1 ####

model = gLASE(d,d, gd_steps)
model.to(device)

### Initialization
model.gd.lin1.weight.data = 4*0.001*torch.eye(d).to(device)
model.gd.lin2.weight.data = 4*0.001*torch.eye(d).to(device)
model.gd.lin1.weight.requires_grad = False
model.gd.lin2.weight.requires_grad = False
model.gd.Q.data = torch.nn.init.xavier_uniform_(model.gd.Q).to(device)
model.gd.Q.requires_grad = True
print(model.gd.Q.data)

# Regularization function
def custom_regularization(x):
    return ((x.abs() - 1)**2).sum()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=5, min_delta=0)

lambda_reg = 0.01

train_loss_epoch=[]
val_loss_epoch=[]
min_val_loss = np.inf

start = time.time()

for epoch in range(epochs):
    ## Train
    train_loss_step=[]
    model.train()
    train_loop = tqdm(train_loader)
    for i, batch in enumerate(train_loop):  
        batch.to(device) 
        optimizer.zero_grad()
        A = to_dense_adj(batch.edge_index).squeeze(0).to('cpu').numpy()
        Ur, Sr, Vrt = rsvd(A,d,2,2)
        Xrsvd = Ur[:,0:d].dot(np.diag(np.sqrt(Sr[0:d])))
        x = torch.Tensor(Xrsvd).to(device)
        
        out = model(x, batch.edge_index, edge_index_2)
        I_pq = torch.diag(model.gd.Q[0])
        loss = torch.norm((out@I_pq@out.T - to_dense_adj(batch.edge_index).squeeze(0))) + lambda_reg * custom_regularization(model.gd.Q[0])
        loss.backward() 
        optimizer.step() 
        
        # if epoch % 100 == 0:
        #     print(f'Train Loss: {loss}')
        #     print(I_pq)
        
        train_loss_step.append(loss.detach().to('cpu').numpy())
        train_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        train_loop.set_postfix(loss=loss)

    train_loss_epoch.append(np.array(train_loss_step).mean())
        
    ## Validation
    val_loss_step=[] 
    model.eval()
    val_loop = tqdm(val_loader)
    for i, batch in enumerate(val_loop):
        batch.to(device)      
        A = to_dense_adj(batch.edge_index).squeeze(0).to('cpu').numpy()
        Ur, Sr, Vrt = rsvd(A,d,2,2)
        Xrsvd = Ur[:,0:d].dot(np.diag(np.sqrt(Sr[0:d])))
        x = torch.Tensor(Xrsvd).to(device)
        
        out = model(x, batch.edge_index, edge_index_2)
        I_pq = torch.diag(model.gd.Q[0])
        loss = torch.norm((out@I_pq@out.T - to_dense_adj(batch.edge_index).squeeze(0))) + lambda_reg * custom_regularization(model.gd.Q[0])

        val_loss_step.append(loss.detach().to('cpu').numpy())
        val_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        val_loop.set_postfix(loss=loss)

    val_loss_epoch.append(np.array(val_loss_step).mean())

    if val_loss_epoch[epoch] < min_val_loss:
        min_val_loss = val_loss_epoch[epoch]
        print("Best model updated")
        print("Val loss: ", min_val_loss)
        print(I_pq)
        
    if early_stopper.early_stop(val_loss_epoch[epoch]):    
        optimal_epoch = np.argmin(val_loss_epoch)
        print("Optimal epoch: ", optimal_epoch)         
        break    
    
stop = time.time()
print(f"Training time: {stop - start}s")

# # #### PHASE 2 ####
# # lr=1e-2
# # gd_steps = 5
# # epochs = 10000

# # edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
# # mask = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)

Q = torch.sign(torch.diag(model.gd.Q[0]).detach())
# # # Q = torch.Tensor([[1,0],[0,-1]]).to(device)
# # print('Starting PHASE 2 training')
print('Q=',Q)

# # model2 = gLASE_w(d,d, gd_steps)
# # model2.to(device)    

# # ## Initialization
# # for step in range(gd_steps):
# #     model2.gd[step].lin1.weight.data = torch.nn.init.xavier_uniform_(model2.gd[step].lin1.weight)*lr
# #     model2.gd[step].lin2.weight.data = torch.nn.init.xavier_uniform_(model2.gd[step].lin2.weight)*lr

# # optimizer = torch.optim.Adam(model2.parameters(), lr=lr)
# # early_stopper = EarlyStopper(patience=10, min_delta=0)

# # train_loss_epoch=[]
# # val_loss_epoch=[]
# # min_val_loss = np.inf

# # for epoch in range(epochs):
# #     # Train
# #     train_loss_step=[]
# #     model2.train()
# #     train_loop = tqdm(train_loader)
# #     for i, batch in enumerate(train_loop):  
# #         batch.to(device) 
# #         optimizer.zero_grad()
# #         out = model2(batch.x, batch.edge_index, edge_index_2, Q, mask)
# #         loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index).squeeze(0)))
# #         loss.backward() 
# #         optimizer.step() 
        
# #         train_loss_step.append(loss.detach().to('cpu').numpy())
# #         train_loop.set_description(f"Epoch [{epoch}/{epochs}]")
# #         train_loop.set_postfix(loss=loss)

# #     train_loss_epoch.append(np.array(train_loss_step).mean())
        
# #     # Validation
# #     val_loss_step=[] 
# #     model2.eval()
# #     val_loop = tqdm(val_loader)
# #     for i, batch in enumerate(val_loop):
# #         batch.to(device)      
# #         out = model2(batch.x, batch.edge_index, edge_index_2, Q, mask)
# #         loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index).squeeze(0)))

# #         val_loss_step.append(loss.detach().to('cpu').numpy())
# #         val_loop.set_description(f"Epoch [{epoch}/{epochs}]")
# #         val_loop.set_postfix(loss=loss)

# #     val_loss_epoch.append(np.array(val_loss_step).mean())

# #     if val_loss_epoch[epoch] < min_val_loss:
# #         torch.save(model2.state_dict(), MODEL_FILE)
# #         min_val_loss = val_loss_epoch[epoch]
# #         print("Best model updated")
# #         print("Val loss: ", min_val_loss)

# #     if early_stopper.early_stop(val_loss_epoch[epoch]):    
# #         optimal_epoch = np.argmin(val_loss_epoch)
# #         print("Optimal epoch: ", optimal_epoch)         
# #         break    

# stop = time.time()
# print(f"Training time: {stop - start}s")