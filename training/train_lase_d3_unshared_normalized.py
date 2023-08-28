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
import pickle

from graspologic.embed import AdjacencySpectralEmbed 

num_nodes = 150
# n_components = 2
d = 3
device = 'cpu'
epochs = 10000
lr=1e-3
gd_steps = 5
MODEL_FILE='../saved_models/lase_unshared_d2_normalized_fede_v2.pt'

print(gd_steps)
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

# sns.heatmap(p,vmin=0,vmax=1)
# plt.show()
# plt.title('P')
# sns.heatmap(Xhats@Xhats.T)
# plt.show()    
# plt.title('ASE')
    

model = GD_Unroll(d,d, gd_steps)
model.to(device)


for i in range(gd_steps):
    # Initialize weights GCN
    getattr(model.gd,'module_'+str(i)).gcn.lins[0].weight.data = torch.eye(d).to(device)
    getattr(model.gd,'module_'+str(i)).gcn.lins[1].weight.data = lr*torch.nn.init.xavier_uniform_(getattr(model.gd,'module_'+str(i)).gcn.lins[1].weight).to(device)
    
    # Initialize weights TransformerConv
    getattr(model.gd,'module_'+str(i)).gat.lin2.weight.data = lr*torch.nn.init.xavier_uniform_(getattr(model.gd,'module_'+str(i)).gat.lin2.weight).to(device)
    getattr(model.gd,'module_'+str(i)).gat.lin3.weight.data = torch.eye(d).to(device)
    getattr(model.gd,'module_'+str(i)).gat.lin4.weight.data = torch.eye(d).to(device)
    
    # Freeze weights
    getattr(model.gd,'module_'+str(i)).gcn.lins[0].weight.requires_grad = False
    getattr(model.gd,'module_'+str(i)).gat.lin3.weight.requires_grad = False
    getattr(model.gd,'module_'+str(i)).gat.lin4.weight.requires_grad = False

# for param in model.parameters():
#     print(param)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    # Train
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, edge_index_2)
    loss = torch.norm((out@out.T - to_dense_adj(edge_index).squeeze(0)))
    
    loss.backward() 
    optimizer.step() 

    if epoch % 500 == 0:
        print(f'Train Loss: {loss}')
        # sns.heatmap((out@out.T).detach().numpy())
        # plt.title('LASE unshared - epoch: '+str(epoch))
        # plt.show()

torch.save(model.state_dict(), MODEL_FILE)

with open('./data/edge_index_d3_v2.pkl', 'wb') as f:
    pickle.dump(edge_index, f)