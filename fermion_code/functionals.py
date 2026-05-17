import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from numpy.linalg import eigvalsh


def onsite_sq_ampl(v):
    """
    Returns squared amplitude on each site, given the wavefunction.
    
    Arguments:
        v: wavefunction vector
    
    Returns:
        Array of squared amplitudes
    """
    return np.abs(v * np.conj(v)).astype(float)


def graph_dist_sq_ampl(A, n0):
    """
    Returns a function that, given a wavefunction, yields
    the probability that the particle lie at any graph
    distance from a given node.
    
    Arguments:
        A: adjacency matrix
        n0: the node from which the distances are computed
    
    Returns:
        Function that computes squared amplitude by graph distance
    """
    N = A.shape[0]
    max_dist = max(N - n0, n0 - 1)
    
    # Compute all-pairs shortest paths
    dists = floyd_warshall(A, directed=False)
    node_dists = dists[n0, :].astype(int)
    
    def onsite_sq_ampl_graph_dist(v):
        """Compute squared amplitude grouped by graph distance"""
        output = np.zeros(max_dist + 1)
        amplitudes = onsite_sq_ampl(v)
        
        for i, ampl in enumerate(amplitudes):
            dist = node_dists[i]
            if 0 <= dist <= N:
                output[dist] += ampl
        
        return output
    
    return onsite_sq_ampl_graph_dist


def IPR(v):
    """
    Inverse Participation Ratio.
    
    Arguments:
        v: wavefunction vector
    
    Returns:
        IPR value
    """
    return np.sum(np.abs(v)**4)


def ent_entropy_side_half(v):
    """
    Returns the entanglement entropy between the
    right and left halves of the system, in the single-
    particle case. The input is the wavefunction.
    
    Arguments:
        v: wavefunction vector
    
    Returns:
        Entanglement entropy
    """
    N = len(v)
    half_idx = int(np.floor(N / 2))
    p = np.sum(np.abs(v[:half_idx])**2)
    return compute_entropy(p)


def ent_entropy_middle_half(v):
    """
    Returns the entanglement entropy between the
    sites [N/4, N*3/4] and the rest of the system,
    in the single-particle case. The input is the
    wavefunction.
    
    Arguments:
        v: wavefunction vector
    
    Returns:
        Entanglement entropy
    """
    N = len(v)
    quarter_idx = int(np.floor(N / 4))
    p = (np.sum(np.abs(v[:quarter_idx])**2) + 
         np.sum(np.abs(v[-quarter_idx:])**2))
    return compute_entropy(p)


def compute_entropy(p):
    """
    Returns the Shannon entropy of the probability
    vector (p, 1-p).
    
    Arguments:
        p: probability value
    
    Returns:
        Shannon entropy
    """
    p = np.abs(p)
    p = min(1.0, p)
    
    if p == 0 or p == 1:
        return 0.0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def entropy_of_vector(v):
    """
    Returns the Shannon entropy of a probability vector.
    
    Arguments:
        v: probability vector
    
    Returns:
        Shannon entropy
    """
    v = np.abs(v)
    v = np.minimum(v, 1.0)
    
    entropy = 0.0
    for x in v:
        if x > 0:
            entropy -= x * np.log2(x)
    
    return entropy


def fermions_ent_entropy(Gamma, subsyst_idcs):
    """
    Returns the entanglement entropy for fermionic Gaussian
    states described by a correlation matrix Γ. The partition
    is between the subsystem including the sites within
    'subsyst_idcs' and the rest of the system.
    
    Arguments:
        Gamma: correlation matrix
        subsyst_idcs: indices of subsystem sites
    
    Returns:
        Entanglement entropy
    """
    # Extract subsystem correlation matrix
    subsyst_idcs = np.array(subsyst_idcs)
    Gamma_sub = Gamma[np.ix_(subsyst_idcs, subsyst_idcs)]
    
    # Compute eigenvalues
    lambdas = eigvalsh(Gamma_sub)
    
    # Sum entropies
    return np.sum([compute_entropy(lam) for lam in lambdas])


def fermions_n(Gamma, subsyst_idcs):
    """
    Returns the expected number of excitations in the
    subsystem made of the sites within 'subsyst_idcs',
    for a fermionic Gaussian state. The input is the correlation
    matrix Γ.
    
    Arguments:
        Gamma: correlation matrix
        subsyst_idcs: indices of subsystem sites
    
    Returns:
        Expected particle number
    """
    subsyst_idcs = np.array(subsyst_idcs)
    return np.sum(np.real(np.diag(Gamma)[subsyst_idcs]))


def fermions_ent_entropy_side_half(Gamma):
    """
    Entanglement entropy between left and right halves
    for fermionic Gaussian states.
    
    Arguments:
        Gamma: correlation matrix
    
    Returns:
        Entanglement entropy
    """
    N = Gamma.shape[0]
    half_idx = int(np.floor(N / 2))
    return fermions_ent_entropy(Gamma, range(half_idx))


def fermions_ent_entropy_even(Gamma):
    """
    Entanglement entropy between even and odd sites
    for fermionic Gaussian states.
    
    Arguments:
        Gamma: correlation matrix
    
    Returns:
        Entanglement entropy
    """
    N = Gamma.shape[0]
    return fermions_ent_entropy(Gamma, range(1, N, 2))


def fermions_n_second_half(Gamma):
    """
    Expected particle number in second half of system.
    
    Arguments:
        Gamma: correlation matrix
    
    Returns:
        Expected particle number
    """
    N = Gamma.shape[0]
    half_idx = int(np.floor(N / 2))
    return fermions_n(Gamma, range(half_idx, N))


def fermions_n_even(Gamma):
    """
    Expected particle number on even sites.
    
    Arguments:
        Gamma: correlation matrix
    
    Returns:
        Expected particle number
    """
    N = Gamma.shape[0]
    return fermions_n(Gamma, range(1, N, 2))
