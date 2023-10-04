import numpy.linalg as la
import scipy
import numpy as np

def rsvd(A,r,q,p):
    """
    randomSVD: Implemenation of a random SVD method.
    Used to compute an approximation of the SVD.    

    Parameters
    ----------
    A : matrix to decompose
    r : number of singular values to keep
    q : power iteration parameter (q=1 or q=2 may be enough)
    p : oversampling factor


    Returns
    -------
    U, S, V^T, approximate SVD decomposition of A
    """

    ny = A.shape[1]
    P = np.random.randn(ny,r+p)
    Z = A @ P
    for k in range(q):
        Z = A @ (A.T @ Z)
        
    Q, R = la.qr(Z,mode='reduced')
    Y = Q.T @ A
    UY, S, VT = scipy.linalg.svd(Y)
    U = Q @ UY
    return U, S, VT