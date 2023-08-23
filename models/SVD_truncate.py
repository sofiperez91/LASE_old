import numpy as np
import scipy.linalg

def embed_scipy(A, K):
    UA, SA, VAt = scipy.sparse.linalg.svds(A,K)
    XA = UA[:,0:K].dot(np.diag(np.sqrt(SA[0:K])))    
    return XA