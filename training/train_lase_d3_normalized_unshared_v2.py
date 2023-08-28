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

num_nodes = 150
d = 3
device = 'cpu'
epochs = 50000
lr=1e-2
gd_steps = 5
MODEL_FILE='../saved_models/lase_unshared_d3_normalized_v2.pt'
TRAIN_DATA_FILE='../data/sbm3_train.pkl'
VAL_DATA_FILE='../data/sbm3_val.pkl'

n = [int(num_nodes/3), int(num_nodes/3), int(num_nodes/3)]
p = [
     [0.5, 0.1, 0.3],
     [0.1, 0.9, 0.2], 
     [0.3, 0.2, 0.7]
]

edge_index = stochastic_blockmodel_graph(n, p).to(device)
edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous()
x = torch.ones(num_nodes,d).to(device)

adj_matrix = to_dense_adj(edge_index).squeeze(0).numpy()
ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='full')
Xhats = ase.fit_transform(adj_matrix)
print("best error: "+str(np.linalg.norm(Xhats@Xhats.T-adj_matrix)))

plt.figure(1)
sns.heatmap(p,vmin=0,vmax=1)
plt.title('P')
plt.show(block=False)
plt.pause(0.001)

plt.figure(2)
sns.heatmap(Xhats@Xhats.T)
plt.title('ASE')
plt.show(block=False)    
plt.pause(0.001)
    

model = GD_Unroll(d,d, gd_steps)
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
early_stopper = EarlyStopper(patience=10, min_delta=0)

with open(TRAIN_DATA_FILE, 'rb') as f:
    df_train = pickle.load(f)
df_train = [Data(x = data.x.squeeze(0), edge_index = data.edge_index, edge_index_2 = edge_index_2) for data in df_train]

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)
df_val = [Data(x = data.x.squeeze(0), edge_index = data.edge_index, edge_index_2 = edge_index_2) for data in df_val]

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
        out = model(x, batch.edge_index, edge_index_2)
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
        out = model(x, batch.edge_index, edge_index_2)
        loss = torch.norm((out@out.T - to_dense_adj(batch.edge_index).squeeze(0)))

        val_loss_step.append(loss.detach().to('cpu').numpy())
        val_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        val_loop.set_postfix(loss=loss)

    val_loss_epoch.append(np.array(val_loss_step).mean())

    if val_loss_epoch[epoch] < min_val_loss:
        torch.save(model.state_dict(), MODEL_FILE)
        min_val_loss = val_loss_epoch[epoch]
        print("Best model updated")

    if early_stopper.early_stop(val_loss_epoch[epoch]):    
        optimal_epoch = np.argmin(val_loss_epoch)
        print("Optimal epoch: ", optimal_epoch)         
        break



model.load_state_dict(torch.load(MODEL_FILE))
model.to(device)
model.eval()

out = model(x, edge_index, edge_index_2)
loss = torch.norm((out@out.T - to_dense_adj(edge_index).squeeze(0)))

plt.figure(3)
sns.heatmap((out@out.T).detach().numpy())
plt.show()
plt.title('LASE unshared')
