import torch_geometric as pyg
import torch
from torch_geometric.utils import to_dense_adj



def RDPG_cost(x, A):
    
    return 0.5*torch.norm(x@x.T - A)**2

def RPDG_gradient(x, A):
    
    return 4*(x@x.T-A)@x


def RDPG_GD_fixstep(x, edge_index, tol=1e-3, alpha=0.001, max_iter=10000):

    xd = x
    A = to_dense_adj(edge_index, max_num_nodes=x.shape[0]).squeeze()
    d = -RPDG_gradient(x, A)
    k=0

    while (torch.norm(d) > tol) & (k<max_iter):

        xd = xd+alpha*d
        d = -RPDG_gradient(xd, A)
        k=k+1

    cost = RDPG_cost(xd, A)

    return xd, cost, k 


def RDPG_GD_Armijo(x, edge_index, tol=1e-3, max_iter=10000):

    A = to_dense_adj(edge_index).squeeze()
    b=0.3; sigma=0.1 # Armijo parameters
    t = 0.1
    xd=x
    k=0
    last_jump=1
    d = -RPDG_gradient(xd, A)
    tol = tol*(torch.norm(d))

    while (torch.norm(d) > tol) & (last_jump > 1e-16) & (k<max_iter):

        # Armijo 
        while (RDPG_cost(xd+t*d,A) > RDPG_cost(xd,A) - sigma*t*torch.norm(d)**2):
            t=b*t
            
        xd = xd+t*d
        last_jump = sigma*t*torch.norm(d)**2
        t=t/(b)
        k=k+1
        d = -RPDG_gradient(xd, A)
    
    return xd, RDPG_cost(xd,A), k