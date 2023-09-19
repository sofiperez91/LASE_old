import numpy as np
import scipy.linalg
from torch_geometric.utils import to_dense_adj

def embed_scipy(edge_index, K):
    A = to_dense_adj(edge_index).squeeze().numpy()
    UA, SA, VAt = scipy.sparse.linalg.svds(A,K)
    XA = UA[:,0:K].dot(np.diag(np.sqrt(SA[0:K])))    
    return XA