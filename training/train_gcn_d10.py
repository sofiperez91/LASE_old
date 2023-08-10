import pickle
import torch
from torch import nn
import numpy as np
from GD_GCN import GCN
from torch_geometric.loader import DataLoader
import torch_geometric as pyg
from tqdm import tqdm


TRAIN_DATA_FOLDER = './data/df_train_d10_2.pkl'
VAL_DATA_FOLDER = './data/df_val_d10_2.pkl'

TRAIN_LOSS_FILE = './saved_models/loss/gcn_d10_train_full_.pkl'
VAL_LOSS_FILE = './saved_models/loss/gcn_d10_val_full_.pkl'
BEST_MODEL_FILE = './saved_models/gcn_d10_full.pt'
CURRENT_MODEL_FILE = './saved_models/gcn_d10_full_current.pt'
FILE_NAME = './saved_models/gcn_d10_full_'

d= 10
gd_steps = 20
in_channels = 10
out_channels = 10

epochs = 200

lr = 1e-7

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
model = GCN(in_channels,out_channels, gd_steps)
model.to(device)

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
        num_nodes = pyg.utils.to_dense_adj(batch.edge_index).shape[2]
        #M = (torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)).to(device)
        M = torch.ones((num_nodes, num_nodes)).to(device)
        A = pyg.utils.to_dense_adj(batch.edge_index)*M
        out = model(batch.x, A[0].nonzero().t().contiguous())
            
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
        num_nodes = pyg.utils.to_dense_adj(batch.edge_index).shape[2]
        # M = (torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)).to(device)
        M = torch.ones((num_nodes, num_nodes)).to(device)
        A = pyg.utils.to_dense_adj(batch.edge_index)*M
        out = model(batch.x, A[0].nonzero().t().contiguous())

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



