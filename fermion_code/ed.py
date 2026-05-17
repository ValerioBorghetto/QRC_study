import numpy as np
from numpy.linalg import eigh


def evolve_with_ED(A, ts, n0, functional=None, J=1.0):
    """
    Computes a certain functional of the wavefunction (in the
    position basis) for multiple times. Assumes that the
    Hamiltonian is given by the adjacency matrix, multiplied
    by a factor J, and that there is a single particle hopping
    on a graph.
    
    Arguments:
        A: adjacency matrix of the graph (numpy array)
        ts: times at which the functional should be evaluated (list or array)
        n0: initial site occupied by the particle (int)
        functional: function to apply to wavefunction (defaults to identity)
        J: energy scale associated with the hopping term (float)
    
    Returns:
        List of functional evaluations at each time
    """
    if functional is None:
        functional = lambda x: x
    
    # Eigendecomposition of adjacency matrix
    energies, eigstates = eigh(A)
    
    # Initial state in energy basis
    n0_en = eigstates[n0, :]
    
    # Evolve for each time
    results = []
    for t in ts:
        # Time evolution operator in energy basis
        phase_factors = np.exp(-1j * J * t * energies)
        # Evolve and transform back to position basis
        psi_t = eigstates @ np.diag(phase_factors) @ n0_en
        results.append(functional(psi_t))
    
    return results
