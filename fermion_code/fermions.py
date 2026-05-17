import numpy as np
from numpy.linalg import eigh


def evolve_fermions(A, initial_sites, ts, functionals=None):
    """
    Evaluates several functionals of the correlation matrix
    for fermionic Gaussian states at several times, assuming
    a hopping Hamiltonian on a graph and that the initial state
    is obtained from the vacuum by applying creation operators on
    various sites (in the right order).
    
    Arguments:
        A: adjacency matrix (numpy array)
        initial_sites: list of the sites initially occupied
        ts: times at which the functional should be evaluated
        functionals: list of functions to apply (defaults to identity)
    
    Returns:
        List of lists, where each inner list contains the functional
        evaluations for a given time
    """
    if functionals is None:
        functionals = [lambda x: x]
    
    N = A.shape[0]
    
    # Eigendecomposition of adjacency matrix
    epsilons, vecs = eigh(A)
    
    # Set up correlations at t=0
    # Γ[i,j] = 1 if site i is occupied, 0 otherwise (on diagonal)
    Gamma = np.diag([1.0 if i in initial_sites else 0.0 for i in range(N)])
    
    # Build the fermionic quadratic hamiltonian
    # \hat H = ∑ J_{i j} a_i^\dagger a_j
    # becomes
    # \hat H = ∑ 1/2 J_{i j} (a_i^\dagger a_j - a_i a_j^\dagger)
    # H here is multiplied by a factor 2
    
    results = []
    for i_t, t in enumerate(ts):
        # Time evolution in the single-particle basis
        U = vecs @ np.diag(np.exp(-1j * epsilons * t)) @ vecs.T.conj()
        
        # Evolve correlation matrix
        Gamma_out = U.T.conj() @ Gamma @ U
        
        # Apply all functionals
        results.append([f(Gamma_out) for f in functionals])
    
    return results
