import numpy as np
from numpy.linalg import eigh

from fermion_code.functionals import (
    ent_entropy_side_half,
    fermions_ent_entropy_side_half,
)


def _unitary_from_H(H, dt):
    energies, vecs = eigh(H)
    U = vecs @ np.diag(np.exp(-1j * energies * dt)) @ vecs.T.conj()
    return U


def build_H_with_input(A, u_t, mask, epsilon, J=1.0):
    H = J * A.astype(float).copy()
    H += epsilon * u_t * np.diag(mask)
    return H


def run_qrc(
    A,
    input_signal,
    dt=0.5,
    epsilon=0.5,
    gamma=0.05,
    J=1.0,
    T_washout=150,
    seed_mask=99,
    use_offdiag=False,
    compute_single_entropy=False,
    compute_fermionic_entropy=False,
    n0_single=None,
    n_fermions=None,
):
    """
    Parametro aggiuntivo:
        n_fermions: int o None.
            None  → half-filling (N//2 particelle, comportamento originale).
            int n → n particelle. Il target del bath viene riscalato in modo
                    che Tr(Gamma_target) = n a ogni step (Opzione A: filling
                    medio conservato). Fisicamente corrisponde a un bath che
                    modula le occupazioni mantenendo la densità media fissata.
    """
    N = A.shape[0]
    T_total = len(input_signal)
    T_valid = T_total - T_washout

    if T_valid <= 0:
        raise ValueError(
            f"T_washout={T_washout} >= T_total={T_total}. "
            "Riduci T_washout o allunga il segnale di input."
        )

    if n_fermions is None:
        n_fermions = N // 2

    if not (1 <= n_fermions <= N):
        raise ValueError(f"n_fermions={n_fermions} deve essere in [1, {N}].")

    rng = np.random.default_rng(seed_mask)
    mask = rng.uniform(-1, 1, size=N)

    Gamma = np.diag(
        [1.0 if i < n_fermions else 0.0 for i in range(N)]
    ).astype(complex)

    if n0_single is None:
        n0_single = N // 4

    psi = None
    if compute_single_entropy:
        psi = np.zeros(N, dtype=complex)
        psi[n0_single] = 1.0

    triu_idx = np.triu_indices(N, k=1)
    D = N + (N * (N - 1) // 2 if use_offdiag else 0)
    features = np.zeros((T_valid, D))

    S_single_trace = [] if compute_single_entropy else None
    S_fermionic_trace = [] if compute_fermionic_entropy else None

    for t, u_t in enumerate(input_signal):
        H_t = build_H_with_input(A, u_t, mask, epsilon, J)
        U = _unitary_from_H(H_t, dt)

        # Schema "reset-poi-evolvi" derivabile da una Lindblad quadratica fermionica
        # (Prosen 2008, Martinez-Pena et al. PRL 2021, Sannia et al. Quantum 2022).
        #
        # Ordine corretto:
        #   1. Reset parziale verso Gamma_target(u_t) con rate gamma
        #   2. Evoluzione unitaria sul risultato del reset
        #
        # Gamma_target e' diagonale con occupazioni sigma(epsilon * u_t * xi_i),
        # dove sigma(x) = (1 + tanh(x)) / 2 mappa R -> (0,1), garantendo che
        # Gamma_target sia sempre uno stato gaussiano fermionico fisico (0 <= Gamma <= I)
        # senza bisogno di clip o diagonalizzazione aggiuntiva.
        occupations_target = 0.5 * (1.0 + np.tanh(epsilon * u_t * mask))
        # Riscala a n_fermions particelle (conservazione del filling medio).
        # Per n_fermions = N//2 e' quasi un no-op (la somma e' gia' ~N/2).
        occ_sum = occupations_target.sum()
        if occ_sum > 1e-12:
            occupations_target = occupations_target * (n_fermions / occ_sum)
        Gamma_target = np.diag(occupations_target).astype(complex)

        Gamma_reset = (1.0 - gamma) * Gamma + gamma * Gamma_target
        Gamma = U.T.conj() @ Gamma_reset @ U

        if compute_single_entropy:
            psi = U @ psi
            S_single_trace.append(ent_entropy_side_half(psi))

        if compute_fermionic_entropy:
            S_fermionic_trace.append(fermions_ent_entropy_side_half(Gamma))

        if t >= T_washout:
            idx = t - T_washout
            features[idx, :N] = np.real(np.diag(Gamma))

            if use_offdiag:
                features[idx, N:] = np.real(Gamma[triu_idx])

    diagnostics = {}
    # Nota: S_single_trace e S_fermionic_trace hanno lunghezza T_total (tutti i passi,
    # incluso il washout). Il chiamante deve tagliare con lo stesso T_washout
    # usato qui per allinearle con features (che ha già solo T_valid campioni).
    # Il T_washout usato per il taglio esterno è incluso nei diagnostics per
    # evitare disallineamenti silenziosi.
    diagnostics["T_washout"] = T_washout

    if compute_single_entropy:
        S_single_trace = np.array(S_single_trace)
        assert len(S_single_trace) == T_total, (
            f"S_single_trace ha {len(S_single_trace)} elementi, attesi {T_total}"
        )
        diagnostics["S_single_trace"] = S_single_trace
        diagnostics["dS_single_dt"] = np.gradient(S_single_trace, dt)

    if compute_fermionic_entropy:
        S_fermionic_trace = np.array(S_fermionic_trace)
        assert len(S_fermionic_trace) == T_total, (
            f"S_fermionic_trace ha {len(S_fermionic_trace)} elementi, attesi {T_total}"
        )
        diagnostics["S_fermionic_trace"] = S_fermionic_trace
        diagnostics["dS_fermionic_dt"] = np.gradient(S_fermionic_trace, dt)

    return features, Gamma, diagnostics