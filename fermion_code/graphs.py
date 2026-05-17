import numpy as np
from scipy.linalg import block_diag


def build_graph(c, alpha, N):
    """
    Builds a graph of N nodes, randomly sampled according to
    the distribution defined in Gori et al., Phys. Rev. E (2017).
    
    Arguments:
        c: coupling constant
        alpha: power-law exponent
        N: number of nodes
    
    Returns:
        Symmetric adjacency matrix (numpy array)
    """
    A = np.zeros((N, N), dtype=int)
    
    for i in range(N - 1):
        for j in range(i + 1, N):
            # Connection probability ~ c / (j - i)^alpha
            if np.random.rand() < c / ((j - i) ** alpha):
                A[i, j] = 1
    
    # Make symmetric
    A = A + A.T
    
    return A


def kac_normalization(c, alpha, N):
    """
    Compute Kac normalization factor.
    
    Arguments:
        c: coupling constant
        alpha: power-law exponent
        N: number of nodes
    
    Returns:
        Normalization factor
    """
    return 1.0 / c / np.sum([r**(-alpha) for r in range(1, N + 1)])


def fully_connected_graph(N):
    """
    Build a fully connected graph (complete graph).
    
    Arguments:
        N: number of nodes
    
    Returns:
        Adjacency matrix (numpy array)
    """
    return np.ones((N, N), dtype=int) - np.eye(N, dtype=int)


def ising_1d_obc(N):
    """
    Build 1D Ising chain with open boundary conditions.
    Returns a symmetric tridiagonal matrix with zeros on diagonal
    and ones on off-diagonals.
    
    Arguments:
        N: number of sites
    
    Returns:
        Symmetric tridiagonal matrix (numpy array)
    """
    # Create tridiagonal matrix with zeros on diagonal, ones on off-diagonals
    A = np.zeros((N, N), dtype=int)
    for i in range(N - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    
    return A
