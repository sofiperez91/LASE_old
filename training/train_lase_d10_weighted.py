import sys
sys.path.append("../")

import pickle
import torch
from torch import nn
import numpy as np
from models.RDPG_GD_Unroll_weighted import GD_Unroll
from torch_geometric.loader import DataLoader
import torch_geometric as pyg
from tqdm import tqdm
from torch_geometric.data import Data


TRAIN_DATA_FOLDER = '../data/df_train_d10_2.pkl'
VAL_DATA_FOLDER = '../data/df_val_d10_2.pkl'

TRAIN_LOSS_FILE = '../saved_models/loss/lase_w_ini_fix_d10_train_M_10layers.pkl'
VAL_LOSS_FILE = '../saved_models/loss/lase_w_ini_fix_d10_val_M_10layers.pkl'
BEST_MODEL_FILE = '../saved_models/lase_w_ini_fix_d10_M_10layers.pt'
CURRENT_MODEL_FILE = '../saved_models/lase_w_ini_fix_d10_M_10layers_current.pt'
FILE_NAME = '../saved_models/lase_w_ini_fix_d10_train_M_10layers_'

num_nodes = 500
d = 10
gd_steps = 50
in_channels = 10
out_channels = 10
epochs = 200
lr = 1e-8
device = 'cuda'

#M = torch.ones((num_nodes, num_nodes)).to(device)
M = (torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)).to(device)

## Load data
with open(TRAIN_DATA_FOLDER, 'rb') as f:
    df_train = pickle.load(f)

B = torch.ones([num_nodes,num_nodes],).to('cuda')
edge_index_2=(B*M).nonzero().t().contiguous()

df_train = [Data(x = data.x.squeeze(0), edge_index = data.edge_index, edge_index_2 = edge_index_2) for data in df_train]

print("Loaded train dataloader")
with open(VAL_DATA_FOLDER, 'rb') as f:
    df_val = pickle.load(f)

df_val = [Data(x = data.x.squeeze(0), edge_index = data.edge_index, edge_index_2 = edge_index_2) for data in df_val]

print("Loaded val dataloader")

train_loader=  DataLoader(df_train, batch_size=6, shuffle = True)
val_loader =  DataLoader(df_val, batch_size=4, shuffle = False)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print("Min Val Loss: ", self.min_validation_loss)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience: 
                return True
            print("Min Val Loss: ", self.min_validation_loss)
        return False

model = GD_Unroll(in_channels,out_channels, gd_steps)
model.to(device)

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

train_loss_arr=[]
val_loss_arr=[]
early_stopper = EarlyStopper(patience=5)

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
        
        M = (torch.ones((out.shape[0], out.shape[0])) - torch.eye(out.shape[0])).to(device)
        
        loss = torch.norm((out@out.T - pyg.utils.to_dense_adj(batch.edge_index).squeeze(0))*M)
        loss.backward() 
        optimizer.step() 

        train_loss_step.append(loss.detach().to('cpu').numpy())

        train_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        train_loop.set_postfix(loss=loss)

        # Break if loss is NaN
        if torch.isnan(loss):
            print(loss)
            break
    # Break if loss is NaN
    if torch.isnan(loss):
        print(loss)
        break

    train_loss_arr.append(np.array(train_loss_step).mean())

    with open(TRAIN_LOSS_FILE, 'wb') as f:
        pickle.dump(train_loss_arr, f)


    # Validation
    val_loss_step=[] 
    model.eval()
    val_loop = tqdm(val_loader)
    for i, batch in enumerate(val_loop):
        batch.to(device)      
        
        out = model(batch.x, batch.edge_index, batch.edge_index_2)
        
        M = (torch.ones((out.shape[0], out.shape[0])) - torch.eye(out.shape[0])).to(device)


        loss = torch.norm((out@out.T - pyg.utils.to_dense_adj(batch.edge_index).squeeze())*M)

        val_loss_step.append(loss.detach().to('cpu').numpy())

        val_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        val_loop.set_postfix(loss=loss)

    val_loss_arr.append(np.array(val_loss_step).mean())

    with open(VAL_LOSS_FILE, 'wb') as f:
        pickle.dump(val_loss_arr, f)

    if epoch % 5 == 0:
        torch.save(model.state_dict(), FILE_NAME+str(epoch)+'.pt')
        print("Saved model")

    if val_loss_arr[epoch] < min_val_loss:
        torch.save(model.state_dict(), BEST_MODEL_FILE)
        min_val_loss = val_loss_arr[epoch]
        print("Best model updated")
    

    torch.save(model.state_dict(), CURRENT_MODEL_FILE)
    print("Train Loss: ",train_loss_arr[epoch])
    print("Val Loss: ", val_loss_arr[epoch])
    if early_stopper.early_stop(val_loss_arr[epoch]):    
        optimal_epoch = np.argmin(val_loss_arr)
        print("Optimal epoch: ", optimal_epoch)         
        break

