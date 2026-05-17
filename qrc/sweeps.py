"""
qrc/sweeps.py
-------------
Tutte le funzioni sweep del progetto QRC.

Ogni funzione sweep ha la stessa struttura:
  1. Definisce la griglia di parametri da esplorare
  2. Chiama _run_sweep_over_axis() che parallelizza su (punto, campione)
  3. Ritorna un dict con risultati aggregati pronti per la serializzazione JSON

Sweep disponibili:
  sweep_dt          → alpha x dt          (fissi: c, N, gamma, epsilon, J, filling)
  sweep_c           → c x alpha_fixed     (fissi: dt, N, gamma, epsilon, J, filling)
  sweep_filling     → alpha x nu=n/N      (fissi: dt, c, N, gamma, epsilon, J)
  sweep_eps_gamma   → alpha x (epsilon, gamma)  (fissi: dt, c, N, J, filling)
  sweep_J           → alpha x J           (fissi: dt, c, N, gamma, epsilon, filling)

Tutti i sweep producono le stesse chiavi nei risultati:
  S_inf_static, nmse, mc_total, mc_per_delay,
  S_single_dynamic, dS_single_peak,
  S_fermionic, dS_fermionic_peak
"""

import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import numpy as np
from joblib import Parallel, delayed

from qrc.core import run_one_sample, aggregate_samples


# ---------------------------------------------------------------------------
# Costruzione alphas standard (identica allo sweep originale)
# ---------------------------------------------------------------------------

def default_alphas():
    return np.concatenate([
        np.linspace(0.01, 0.5, 10),
        np.linspace(0.6,  2.5, 10),
        np.linspace(2.8,  6.1,  3),
    ])


# ---------------------------------------------------------------------------
# Motore comune: loop su una lista di "punti" parametrici
# ---------------------------------------------------------------------------

def _run_sweep_over_axis(
    points,           # lista di dict: ogni dict contiene i parametri del punto
    axis_label,       # stringa per il log (es. "alpha", "c", "nu")
    n_samples,
    seed,
    compute_single_entropy,
    compute_fermionic_entropy,
    mc_max_delay,
    save_traces,
    verbose,
    n_jobs,
):
    """
    Per ogni punto in `points`, lancia n_samples run in parallelo e aggrega.
    `points` è una lista di dict con chiavi che matchano gli argomenti di
    run_one_sample (esclusi graph_seed, i_sample, save_traces).

    Ritorna lista di dict aggregati (uno per punto).
    """
    rng = np.random.default_rng(seed)
    all_seeds = {
        (i_pt, i_s): int(rng.integers(0, 2**31))
        for i_pt in range(len(points))
        for i_s in range(n_samples)
    }

    aggregated = []

    for i_pt, params in enumerate(points):
        label_val = params.get(axis_label, i_pt)
        t0 = time.perf_counter()

        if verbose:
            print(f"  [{i_pt+1}/{len(points)}] {axis_label}={label_val} ...", flush=True)

        sample_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_one_sample)(
                graph_seed=all_seeds[(i_pt, i_s)],
                alpha=params["alpha"],
                c=params["c"],
                N=params["N"],
                dt=params["dt"],
                epsilon=params["epsilon"],
                gamma=params["gamma"],
                J=params["J"],
                T_signal=params["T_signal"],
                T_washout=params["T_washout"],
                alpha_ridge=params["alpha_ridge"],
                n_fermions=params["n_fermions"],
                compute_single_entropy=compute_single_entropy,
                compute_fermionic_entropy=compute_fermionic_entropy,
                mc_max_delay=mc_max_delay,
                save_traces=save_traces,
                i_sample=i_s,
            )
            for i_s in range(n_samples)
        )

        agg = aggregate_samples(
            sample_results, compute_single_entropy, compute_fermionic_entropy,
            mc_max_delay, save_traces,
        )
        agg["params"] = params   # includi i parametri del punto per riferimento
        elapsed = time.perf_counter() - t0

        if verbose:
            skip_info = ", ".join(f"{v} {k}" for k, v in agg["n_skipped"].items() if v > 0)
            msg = (
                f"    n_valid={agg['n_valid']}/{n_samples}"
                + (f" [skip: {skip_info}]" if skip_info else "")
                + f"  t={elapsed:.1f}s"
                + f"  S_static={agg['S_inf_static_mean']:.4f}"
                + f"  NMSE={agg['nmse_mean']:.4f}"
                + f"  MC={agg['mc_total_mean']:.4f}"
            )
            if compute_single_entropy:
                msg += f"  S_s={agg.get('S_single_dynamic_mean', float('nan')):.4f}"
            if compute_fermionic_entropy:
                msg += f"  S_f={agg.get('S_fermionic_mean', float('nan')):.4f}"
            print(msg, flush=True)

        aggregated.append(agg)

    return aggregated


def _default_params(overrides):
    """Parametri di default condivisi da tutti gli sweep."""
    base = dict(
        N=100, T_signal=600, T_washout=50, alpha_ridge=1e-4,
        dt=0.5, epsilon=0.5, gamma=0.05, J=1.0, c=1.0,
        alpha=1.0, n_fermions=None,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. sweep_dt: alpha x dt
# ---------------------------------------------------------------------------

def sweep_dt(
    alphas=None,
    dt_values=None,
    c=1.0, N=100, n_samples=20, T_signal=600, T_washout=50,
    epsilon=0.5, gamma=0.05, J=1.0, alpha_ridge=1e-4,
    compute_single_entropy=True, compute_fermionic_entropy=True,
    mc_max_delay=20, save_traces=False, seed=42, verbose=True, n_jobs=-1,
):
    """Sweep su alpha per diversi valori di dt."""
    if alphas is None:
        alphas = default_alphas()
    if dt_values is None:
        dt_values = np.arange(0.1, 1.1, 0.3)   # [0.1, 0.4, 0.7, 1.0]

    results = {}
    for dt in dt_values:
        if verbose:
            print(f"\n{'='*60}\n  sweep_dt: dt={dt:.2f}  ({len(alphas)} alpha)\n{'='*60}")
        t0 = time.perf_counter()

        points = [
            _default_params(dict(alpha=float(a), dt=float(dt), c=c, N=N,
                                 T_signal=T_signal, T_washout=T_washout,
                                 epsilon=epsilon, gamma=gamma, J=J,
                                 alpha_ridge=alpha_ridge))
            for a in alphas
        ]

        agg_list = _run_sweep_over_axis(
            points, axis_label="alpha",
            n_samples=n_samples, seed=seed,
            compute_single_entropy=compute_single_entropy,
            compute_fermionic_entropy=compute_fermionic_entropy,
            mc_max_delay=mc_max_delay, save_traces=save_traces,
            verbose=verbose, n_jobs=n_jobs,
        )

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  Tempo totale dt={dt:.2f}: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        results[f"dt_{dt:.2f}"] = {
            "dt": float(dt),
            "alphas": [float(a) for a in alphas],
            "sweep": agg_list,
        }

    return results


# ---------------------------------------------------------------------------
# 2. sweep_c: c x alpha_fixed (a diversi alpha rappresentativi)
# ---------------------------------------------------------------------------

def sweep_c(
    c_values=None,
    alpha_fixed_values=None,
    N=100, n_samples=20, T_signal=600, T_washout=50,
    dt=0.5, epsilon=0.5, gamma=0.05, J=1.0, alpha_ridge=1e-4,
    compute_single_entropy=True, compute_fermionic_entropy=True,
    mc_max_delay=20, save_traces=False, seed=42, verbose=True, n_jobs=-1,
):
    """
    Sweep su c per diversi alpha fissati.
    alpha_fixed_values: lista di alpha rappresentativi — default [0.3, 2.5, 6.5]
                        (regime long-range, critico, short-range)
    """
    if c_values is None:
        c_values = np.linspace(0.1, 1.0, 10)
    if alpha_fixed_values is None:
        alpha_fixed_values = [0.3, 2.5, 6.5]   # piccolo, medio, grande

    results = {}
    for alpha_fixed in alpha_fixed_values:
        if verbose:
            print(f"\n{'='*60}\n  sweep_c: alpha={alpha_fixed:.2f}  ({len(c_values)} c)\n{'='*60}")
        t0 = time.perf_counter()

        points = [
            _default_params(dict(alpha=float(alpha_fixed), c=float(cv), N=N,
                                 T_signal=T_signal, T_washout=T_washout,
                                 dt=dt, epsilon=epsilon, gamma=gamma, J=J,
                                 alpha_ridge=alpha_ridge))
            for cv in c_values
        ]

        agg_list = _run_sweep_over_axis(
            points, axis_label="c",
            n_samples=n_samples, seed=seed,
            compute_single_entropy=compute_single_entropy,
            compute_fermionic_entropy=compute_fermionic_entropy,
            mc_max_delay=mc_max_delay, save_traces=save_traces,
            verbose=verbose, n_jobs=n_jobs,
        )

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  Tempo totale alpha={alpha_fixed:.2f}: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        results[f"alpha_{alpha_fixed:.2f}"] = {
            "alpha_fixed": float(alpha_fixed),
            "c_values": [float(cv) for cv in c_values],
            "sweep": agg_list,
        }

    return results


# ---------------------------------------------------------------------------
# 3. sweep_filling: alpha x nu=n/N
# ---------------------------------------------------------------------------

def sweep_filling(
    alphas=None,
    nu_values=None,
    c=1.0, N=100, n_samples=20, T_signal=600, T_washout=50,
    dt=0.5, epsilon=0.5, gamma=0.05, J=1.0, alpha_ridge=1e-4,
    compute_single_entropy=True, compute_fermionic_entropy=True,
    mc_max_delay=20, save_traces=False, seed=42, verbose=True, n_jobs=-1,
):
    """
    Sweep su alpha per diversi filling nu = n_fermions/N.
    Per simmetria particella-buca basta esplorare nu in (0, 0.5].
    """
    if alphas is None:
        alphas = default_alphas()
    if nu_values is None:
        # filling 1/N (singola particella), 0.1, 0.2, 0.3, 0.5 (half)
        nu_values = [1/N, 0.1, 0.2, 0.3, 0.5]

    results = {}
    for nu in nu_values:
        n_f = max(1, int(round(nu * N)))
        label = f"nu_{nu:.3f}_n{n_f}"

        if verbose:
            print(f"\n{'='*60}\n  sweep_filling: nu={nu:.3f} (n={n_f})  ({len(alphas)} alpha)\n{'='*60}")
        t0 = time.perf_counter()

        points = [
            _default_params(dict(alpha=float(a), c=c, N=N,
                                 T_signal=T_signal, T_washout=T_washout,
                                 dt=dt, epsilon=epsilon, gamma=gamma, J=J,
                                 alpha_ridge=alpha_ridge,
                                 n_fermions=n_f))
            for a in alphas
        ]

        agg_list = _run_sweep_over_axis(
            points, axis_label="alpha",
            n_samples=n_samples, seed=seed,
            compute_single_entropy=compute_single_entropy,
            compute_fermionic_entropy=compute_fermionic_entropy,
            mc_max_delay=mc_max_delay, save_traces=save_traces,
            verbose=verbose, n_jobs=n_jobs,
        )

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  Tempo totale nu={nu:.3f}: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        results[label] = {
            "nu": float(nu),
            "n_fermions": n_f,
            "alphas": [float(a) for a in alphas],
            "sweep": agg_list,
        }

    return results


# ---------------------------------------------------------------------------
# 4. sweep_eps_gamma: alpha x (epsilon, gamma)
# ---------------------------------------------------------------------------

def sweep_eps_gamma(
    alphas=None,
    epsilon_values=None,
    gamma_values=None,
    c=1.0, N=100, n_samples=20, T_signal=600, T_washout=50,
    dt=0.5, J=1.0, alpha_ridge=1e-4,
    compute_single_entropy=True, compute_fermionic_entropy=True,
    mc_max_delay=20, save_traces=False, seed=42, verbose=True, n_jobs=-1,
):
    """
    Sweep su alpha per ogni combinazione (epsilon, gamma).
    Produce un risultato per ogni coppia (eps, gamma).
    """
    if alphas is None:
        alphas = default_alphas()
    if epsilon_values is None:
        epsilon_values = [0.2, 0.5, 1.0, 2.0]
    if gamma_values is None:
        gamma_values   = [0.01, 0.05, 0.10, 0.20]

    results = {}
    combos = [(eps, gam) for eps in epsilon_values for gam in gamma_values]

    for eps, gam in combos:
        label = f"eps_{eps:.2f}_gamma_{gam:.3f}"

        if verbose:
            print(f"\n{'='*60}\n  sweep_eps_gamma: eps={eps:.2f} gamma={gam:.3f}  ({len(alphas)} alpha)\n{'='*60}")
        t0 = time.perf_counter()

        points = [
            _default_params(dict(alpha=float(a), c=c, N=N,
                                 T_signal=T_signal, T_washout=T_washout,
                                 dt=dt, epsilon=float(eps), gamma=float(gam), J=J,
                                 alpha_ridge=alpha_ridge))
            for a in alphas
        ]

        agg_list = _run_sweep_over_axis(
            points, axis_label="alpha",
            n_samples=n_samples, seed=seed,
            compute_single_entropy=compute_single_entropy,
            compute_fermionic_entropy=compute_fermionic_entropy,
            mc_max_delay=mc_max_delay, save_traces=save_traces,
            verbose=verbose, n_jobs=n_jobs,
        )

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  Tempo eps={eps:.2f} gamma={gam:.3f}: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        results[label] = {
            "epsilon": float(eps),
            "gamma":   float(gam),
            "alphas":  [float(a) for a in alphas],
            "sweep":   agg_list,
        }

    return results


# ---------------------------------------------------------------------------
# 5. sweep_J: alpha x J
# ---------------------------------------------------------------------------

def sweep_J(
    alphas=None,
    J_values=None,
    c=1.0, N=100, n_samples=20, T_signal=600, T_washout=50,
    dt=0.5, epsilon=0.5, gamma=0.05, alpha_ridge=1e-4,
    compute_single_entropy=True, compute_fermionic_entropy=True,
    mc_max_delay=20, save_traces=False, seed=42, verbose=True, n_jobs=-1,
):
    """
    Sweep su alpha per diversi valori di J (scala di energia dell'hopping).
    Nota: J e dt compaiono sempre nel prodotto J*dt nell'esponenziale,
    quindi variare J a dt fisso è equivalente a variare dt a J fisso.
    Tuttavia epsilon entra in modo indipendente, quindi il rapporto
    epsilon/(J*||A||) cambia — questo rende il parametro fisicamente
    non banalmente ridondante con dt.
    """
    if alphas is None:
        alphas = default_alphas()
    if J_values is None:
        J_values = [0.25, 0.5, 1.0, 2.0, 4.0]

    results = {}
    for J in J_values:
        label = f"J_{J:.2f}"

        if verbose:
            print(f"\n{'='*60}\n  sweep_J: J={J:.2f}  ({len(alphas)} alpha)\n{'='*60}")
        t0 = time.perf_counter()

        points = [
            _default_params(dict(alpha=float(a), c=c, N=N,
                                 T_signal=T_signal, T_washout=T_washout,
                                 dt=dt, epsilon=epsilon, gamma=gamma, J=float(J),
                                 alpha_ridge=alpha_ridge))
            for a in alphas
        ]

        agg_list = _run_sweep_over_axis(
            points, axis_label="alpha",
            n_samples=n_samples, seed=seed,
            compute_single_entropy=compute_single_entropy,
            compute_fermionic_entropy=compute_fermionic_entropy,
            mc_max_delay=mc_max_delay, save_traces=save_traces,
            verbose=verbose, n_jobs=n_jobs,
        )

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  Tempo totale J={J:.2f}: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        results[label] = {
            "J":      float(J),
            "alphas": [float(a) for a in alphas],
            "sweep":  agg_list,
        }

    return results
