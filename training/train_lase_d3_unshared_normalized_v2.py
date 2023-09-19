import sys
sys.path.append("../")

import time
import torch
import numpy as np
import torch_geometric as pyg
from torch_geometric.utils import stochastic_blockmodel_graph, to_dense_adj, erdos_renyi_graph
from models.RDPG_GD_Unroll_unshared_normalized_v2 import GD_Unroll
from models.bigbird_attention import big_bird_attention
from models.early_stopper import EarlyStopper
from graspologic.embed import AdjacencySpectralEmbed 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
from tqdm import tqdm
from networkx import watts_strogatz_graph
from torch_geometric.utils.convert import from_networkx

num_nodes = 150
d = 3
device = 'cuda'
epochs = 50000
lr=1e-3
gd_steps = 5
MODEL_FILE='../saved_models/lase_unshared_d3_normalized__WS_v4.pt'
TRAIN_DATA_FILE='./data/sbm3_train.pkl'
VAL_DATA_FILE='./data/sbm3_val.pkl'

n = [50, 50, 50]
p = [
     [0.5, 0.1, 0.3],
     [0.1, 0.9, 0.2], 
     [0.3, 0.2, 0.7]
]

edge_index = stochastic_blockmodel_graph(n, p).to(device)
mask = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
# edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)

x = torch.rand((num_nodes, d)).to(device)

# adj_matrix = to_dense_adj(edge_index.to('cpu')).squeeze(0).numpy()
# ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='full')
# Xhats = ase.fit_transform(adj_matrix)
# best_error = np.linalg.norm(Xhats@Xhats.T-adj_matrix)
# print("best error: ", best_error)

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
df_train = [Data(x = data.x, edge_index = data.edge_index) for data in df_train]

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)
df_val = [Data(x = data.x, edge_index = data.edge_index) for data in df_val]

train_loader=  DataLoader(df_train, batch_size=1, shuffle = True)
val_loader =  DataLoader(df_val, batch_size=1, shuffle = False)

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
        # edge_index_2 = erdos_renyi_graph(num_nodes, 0.5, directed=False).to(device) 
        # edge_index_2 = erdos_renyi_graph(num_nodes, 0.3, directed=False).to(device) 
        edge_index_2 = from_networkx(watts_strogatz_graph(num_nodes, 50, 0.5, seed=None)).edge_index.to(device)
        # edge_index_2 = big_bird_attention(window_size=5, random_prob=0.3, num_nodes= num_nodes).to(device) 
        # edge_index_2 = from_networkx(watts_strogatz_graph(num_nodes, 10, 0.9, seed=None)).edge_index.to(device) 
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, edge_index_2, mask)
        loss = torch.norm((out@out.T - to_dense_adj(batch.edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))
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
        # edge_index_2 = erdos_renyi_graph(num_nodes, 0.5, directed=False).to(device) 
        # edge_index_2 = erdos_renyi_graph(num_nodes, 0.3, directed=False).to(device) 
        edge_index_2 = from_networkx(watts_strogatz_graph(num_nodes, 50, 0.5, seed=None)).edge_index.to(device)
        # edge_index_2 = big_bird_attention(window_size=15, random_prob=0.3, num_nodes= num_nodes).to(device) 
        # edge_index_2 = erdos_renyi_graph(num_nodes, 0.3, directed=False).to(device) 
        #  
        out = model(batch.x, batch.edge_index, edge_index_2, mask)
        loss = torch.norm((out@out.T - to_dense_adj(batch.edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))

        val_loss_step.append(loss.detach().to('cpu').numpy())
        val_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        val_loop.set_postfix(loss=loss)

    val_loss_epoch.append(np.array(val_loss_step).mean())

    if val_loss_epoch[epoch] < min_val_loss:
        torch.save(model.state_dict(), MODEL_FILE)
        min_val_loss = val_loss_epoch[epoch]
        # sample_out = model(x, edge_index, edge_index_2, mask)
        # sample_val_loss = torch.norm((sample_out@sample_out.T - to_dense_adj(edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))
        print("Best model updated")
        print("Val loss Avg: ", min_val_loss)
        # print("Val loss Sample: ", sample_val_loss.detach().to('cpu').numpy())
        # print("Best error Sample: ", best_error)

    if early_stopper.early_stop(val_loss_epoch[epoch]):    
        optimal_epoch = np.argmin(val_loss_epoch)
        print("Optimal epoch: ", optimal_epoch)         
        break

stop = time.time()
print(f"Training time: {stop - start}s")