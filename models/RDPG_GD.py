import torch_geometric as pyg
import torch
from torch_geometric.utils import to_dense_adj
import numpy as np
import scipy



def RDPG_cost(x, A):
    
    return 0.5*torch.norm(x@x.T - A)**2

def RPDG_gradient(x, A):
    
    return 4*(x@x.T-A)@x

def solve_linear_system(A,b,xx):
    try:
        result = scipy.linalg.solve(A,b)
    except:
        result = scipy.sparse.linalg.minres(A,b,xx)[0]    
    return result

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


def coordinate_descent(A,d,X=None,tol=1e-5):

    num_nodes=A.shape[0]
    if X is None:
        X = torch.rand((num_nodes, d))
    else:
        X = X.copy()
    R = X.T@X
    fold = -1
    while (abs((fold - RDPG_cost(X, A))/fold) >tol):
        fold = RDPG_cost(X,A)
        for i in range(num_nodes):
            k=X[i,:][np.newaxis]
            R -= k.T@k
            X[i,:] = torch.Tensor(solve_linear_system(R,(A[i,:]@X).T,X[i,:]))
            k=X[i,:][np.newaxis]
            R += k.T@k

    return X