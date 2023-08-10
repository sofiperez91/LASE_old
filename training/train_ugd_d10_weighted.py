import pickle
import torch
from torch import nn
import numpy as np
from GD_Unroll_weighted import GD_Unroll
from torch_geometric.loader import DataLoader
import torch_geometric as pyg
from tqdm import tqdm


TRAIN_DATA_FOLDER = './data/df_train_d10_2.pkl'
VAL_DATA_FOLDER = './data/df_val_d10_2.pkl'

TRAIN_LOSS_FILE = './saved_models/loss/ugd_w_ini_fix_d10_train_full_M_50layers.pkl'
VAL_LOSS_FILE = './saved_models/loss/ugd_w_ini_fix_d10_val_full_M_50layers.pkl'
BEST_MODEL_FILE = './saved_models/ugd_w_ini_fix_d10_full_M_50layers.pt'
CURRENT_MODEL_FILE = './saved_models/ugd_w_ini_fix_d10_full_current_M_50layers.pt'
FILE_NAME = './saved_models/ugd_w_ini_fix_d10_full_M_50layers_'

num_nodes = 500
d = 10
gd_steps = 50
in_channels = 10
out_channels = 10

epochs = 200

lr = 1e-8

## Load data
with open(TRAIN_DATA_FOLDER, 'rb') as f:
    df_train = pickle.load(f)

print("Loaded train dataloader")
with open(VAL_DATA_FOLDER, 'rb') as f:
    df_val = pickle.load(f)

print("Loaded val dataloader")

train_loader=  DataLoader(df_train, batch_size=1, shuffle = True)
val_loader =  DataLoader(df_val, batch_size=1, shuffle = False)


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

device = 'cuda'
model = GD_Unroll(in_channels,out_channels, gd_steps)
model.to(device)

for step in range(gd_steps):
    model.gd[step].gcn.lins[0].weight.data = torch.eye(d).to(device).to(device)
    model.gd[step].gcn.lins[0].weight.requires_grad = False

    model.gd[step].gat.W_2.data = torch.ones([1,d]).to(device)
    model.gd[step].gat.W_2.requires_grad = False

    model.gd[step].gcn.lins[1].weight.data = torch.nn.init.xavier_uniform_(model.gd[step].gcn.lins[1].weight)*lr
    model.gd[step].gat.W_3.data = torch.nn.init.xavier_uniform_(model.gd[step].gat.W_3)*lr


optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_loss_arr=[]
val_loss_arr=[]
early_stopper = EarlyStopper(patience=5)

min_val_loss = np.inf

#M = torch.ones((num_nodes, num_nodes)).to(device)
M = (torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)).to(device)

for epoch in range(epochs):
    # Train
    train_loss_step=[]
    model.train()
    train_loop = tqdm(train_loader)
    for i, batch in enumerate(train_loop):  
        batch.to(device) 
        optimizer.zero_grad()
        
        A = pyg.utils.to_dense_adj(batch.edge_index)*M
        B = pyg.utils.to_dense_adj(batch.edge_index_2)*M
        out = model(batch.x, A[0].nonzero().t().contiguous(), B[0].nonzero().t().contiguous())
            
        loss = torch.norm((out[0,]@out[0,].T - pyg.utils.to_dense_adj(batch.edge_index).squeeze())*M)
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
        A = pyg.utils.to_dense_adj(batch.edge_index)*M
        B = pyg.utils.to_dense_adj(batch.edge_index_2)*M
        out = model(batch.x, A[0].nonzero().t().contiguous(), B[0].nonzero().t().contiguous())

        loss = torch.norm((out[0,]@out[0,].T - pyg.utils.to_dense_adj(batch.edge_index).squeeze())*M)

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

