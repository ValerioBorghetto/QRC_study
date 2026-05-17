"""
qrc/core.py
-----------
Logica comune a tutti gli sweep:
  - single_graph_run: esegue run_qrc + readout + MC su un singolo grafo/segnale
  - run_one_sample:   worker parallelo (chiamato da joblib)
  - aggregate:        raccoglie lista di risultati campione → medie/std per un alpha

Tutti gli sweep in sweeps.py usano queste funzioni.
"""

import sys
import os

# Aggiunge la root del progetto (parent di qrc/) al path,
# in modo che 'fermion_code', 'reservoir', 'tasks_fixed' siano trovabili
# indipendentemente da dove viene invocato lo script.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.sparse.csgraph import connected_components

from fermion_code.graphs import build_graph
from fermion_code.ed import evolve_with_ED
from fermion_code.functionals import ent_entropy_side_half

from reservoir import run_qrc
from tasks_fixed import (
    generate_narma10,
    evaluate_reservoir,
    is_valid_timeseries,
    is_finite_scalar,
    safe_stderr,
)
from qrc.memory_capacity import compute_mc


# ---------------------------------------------------------------------------
# S_inf_static
# ---------------------------------------------------------------------------

def compute_S_inf_static(A, n0=None, t_inf=1e4, J=1.0):
    N = A.shape[0]
    if n0 is None:
        n0 = N // 4
    ts = [t_inf + m * 100.0 for m in range(1, 21)]
    entropies = evolve_with_ED(A, ts, n0, functional=ent_entropy_side_half, J=J)
    return float(np.mean(entropies))


# ---------------------------------------------------------------------------
# Run singolo campione
# ---------------------------------------------------------------------------

def single_graph_run(
    A,
    u_input,
    target,
    dt=0.5,
    epsilon=0.5,
    gamma=0.05,
    T_washout=50,
    alpha_ridge=1e-4,
    J=1.0,
    n_fermions=None,
    compute_single_entropy=True,
    compute_fermionic_entropy=True,
    mc_max_delay=20,
):
    """
    Esegue run_qrc, readout NARMA-10 e Memory Capacity su un singolo grafo.

    Parametri notevoli rispetto all'originale:
        n_fermions: int o None. Se None usa half-filling (N//2).
                    Se intero, riscala il target del bath a n_fermions particelle
                    (opzione A, conservazione del filling medio).
        mc_max_delay: numero di delay per la Memory Capacity.
    """
    N = A.shape[0]
    S_inf_static = compute_S_inf_static(A, J=J)

    features, Gamma_final, diagnostics = run_qrc(
        A, u_input,
        dt=dt, epsilon=epsilon, gamma=gamma,
        T_washout=T_washout, J=J,
        n_fermions=n_fermions,
        compute_single_entropy=compute_single_entropy,
        compute_fermionic_entropy=compute_fermionic_entropy,
    )

    T_washout_used = diagnostics["T_washout"]
    target_valid = target[T_washout_used:]

    result = evaluate_reservoir(features, target_valid, alpha_ridge=alpha_ridge, use_bias=True)

    mc_result = compute_mc(features, u_input, max_delay=mc_max_delay, alpha_ridge=alpha_ridge)

    T_train = result["T_train"]

    out = {
        "S_inf_static":  S_inf_static,
        "nmse_test":     result["nmse_test"],
        "nmse_train":    result["nmse_train"],
        "mc_total":      mc_result["mc_total"],
        "mc_per_delay":  mc_result["mc_per_delay"],
        "instant_error_norm": (
            (result["y_true_test"] - result["y_pred_test"]) ** 2
            / max(np.var(result["y_true_test"]), 1e-12)
        ),
    }

    if compute_single_entropy:
        S_s = diagnostics["S_single_trace"]
        dS_s = diagnostics["dS_single_dt"]
        S_s_valid  = S_s[T_washout_used:]
        dS_s_valid = dS_s[T_washout_used:]
        out.update({
            "S_single_dynamic_mean": float(np.mean(S_s_valid)),
            "S_single_dynamic_std":  float(np.std(S_s_valid)),
            "dS_single_peak":        float(np.max(np.abs(dS_s_valid))),
            "S_single_trace":        S_s,
            "dS_single_dt":          dS_s,
            "S_single_test":         S_s_valid[T_train:],
            "dS_single_test":        dS_s_valid[T_train:],
        })

    if compute_fermionic_entropy:
        S_f = diagnostics["S_fermionic_trace"]
        dS_f = diagnostics["dS_fermionic_dt"]
        S_f_valid  = S_f[T_washout_used:]
        dS_f_valid = dS_f[T_washout_used:]
        out.update({
            "S_fermionic_mean":      float(np.mean(S_f_valid)),
            "S_fermionic_std":       float(np.std(S_f_valid)),
            "dS_fermionic_peak":     float(np.max(np.abs(dS_f_valid))),
            "S_fermionic_trace":     S_f,
            "dS_fermionic_dt":       dS_f,
            "S_fermionic_test":      S_f_valid[T_train:],
            "dS_fermionic_test":     dS_f_valid[T_train:],
        })

    return out


def run_one_sample(
    graph_seed,
    # parametri grafo
    alpha, c, N,
    # parametri dinamica
    dt, epsilon, gamma, J,
    T_signal, T_washout, alpha_ridge,
    n_fermions,
    # flag entropie
    compute_single_entropy,
    compute_fermionic_entropy,
    mc_max_delay,
    save_traces,
    i_sample,
):
    """
    Worker eseguito da ogni processo joblib.
    Costruisce il grafo, genera il segnale NARMA-10, chiama single_graph_run.
    Ritorna un dict con i risultati o {"skipped": motivo}.
    """
    narma_seed = graph_seed ^ 0xDEAD
    np.random.seed(graph_seed)

    A = build_graph(c, alpha, N)
    n_comp, _ = connected_components(A, directed=False)
    if n_comp > 1:
        return {"skipped": "disconnected"}

    u, y = generate_narma10(T_signal, seed=narma_seed)
    if not is_valid_timeseries(u, y):
        return {"skipped": "narma_invalid"}

    try:
        out = single_graph_run(
            A, u, y,
            dt=dt, epsilon=epsilon, gamma=gamma,
            T_washout=T_washout, alpha_ridge=alpha_ridge, J=J,
            n_fermions=n_fermions,
            compute_single_entropy=compute_single_entropy,
            compute_fermionic_entropy=compute_fermionic_entropy,
            mc_max_delay=mc_max_delay,
        )
    except Exception as e:
        return {"skipped": f"error:{e}"}

    if not is_finite_scalar(out.get("nmse_test", np.nan)):
        return {"skipped": "nmse_nonfinite"}

    result = {
        "skipped":       None,
        "S_inf_static":  out["S_inf_static"],
        "nmse_test":     out["nmse_test"],
        "nmse_train":    out["nmse_train"],
        "mc_total":      out["mc_total"],
        "mc_per_delay":  out["mc_per_delay"].tolist(),
    }

    for key in [
        "S_single_dynamic_mean", "dS_single_peak",
        "S_fermionic_mean", "dS_fermionic_peak",
    ]:
        if key in out:
            result[key] = out[key]

    if save_traces:
        result["trace"] = {
            "alpha": alpha, "sample": i_sample,
            "instant_error_norm": out["instant_error_norm"].tolist(),
        }
        for key in [
            "S_single_trace", "dS_single_dt", "S_single_test", "dS_single_test",
            "S_fermionic_trace", "dS_fermionic_dt", "S_fermionic_test", "dS_fermionic_test",
        ]:
            if key in out:
                v = out[key]
                result["trace"][key] = v.tolist() if hasattr(v, "tolist") else list(v)

    return result


# ---------------------------------------------------------------------------
# Aggregazione di una lista di risultati campione
# ---------------------------------------------------------------------------

def _smean(lst):
    return float(np.mean(lst)) if lst else np.nan

def _sstd(lst):
    return float(safe_stderr(lst)) if lst else np.nan


def aggregate_samples(sample_results, compute_single_entropy, compute_fermionic_entropy,
                      mc_max_delay, save_traces):
    """
    Prende la lista di dict ritornati da run_one_sample per un dato punto
    parametrico e restituisce un dict aggregato (medie, std, liste grezze).
    """
    n_skipped = {}
    S_static_list, nmse_list, nmse_train_list = [], [], []
    mc_total_list, mc_per_delay_list = [], []
    S_single_list, dS_single_list = [], []
    S_ferm_list, dS_ferm_list = [], []
    traces = []

    for r in sample_results:
        if r is None or r.get("skipped"):
            reason = (r or {}).get("skipped", "unknown").split(":")[0]
            n_skipped[reason] = n_skipped.get(reason, 0) + 1
            continue

        S_static_list.append(r["S_inf_static"])
        nmse_list.append(r["nmse_test"])
        nmse_train_list.append(r["nmse_train"])
        mc_total_list.append(r["mc_total"])
        mc_per_delay_list.append(r["mc_per_delay"])

        if compute_single_entropy and "S_single_dynamic_mean" in r:
            S_single_list.append(r["S_single_dynamic_mean"])
            dS_single_list.append(r["dS_single_peak"])
        if compute_fermionic_entropy and "S_fermionic_mean" in r:
            S_ferm_list.append(r["S_fermionic_mean"])
            dS_ferm_list.append(r["dS_fermionic_peak"])
        if save_traces and "trace" in r:
            traces.append(r["trace"])

    mc_per_delay_mean = (
        np.mean(mc_per_delay_list, axis=0).tolist()
        if mc_per_delay_list else [np.nan] * mc_max_delay
    )
    mc_per_delay_std = (
        (np.std(mc_per_delay_list, axis=0) / max(len(mc_per_delay_list)**0.5, 1)).tolist()
        if mc_per_delay_list else [np.nan] * mc_max_delay
    )

    agg = {
        "n_valid":   len(nmse_list),
        "n_skipped": n_skipped,
        # medie e std
        "S_inf_static_mean":  _smean(S_static_list),
        "S_inf_static_std":   _sstd(S_static_list),
        "nmse_mean":          _smean(nmse_list),
        "nmse_std":           _sstd(nmse_list),
        "nmse_train_mean":    _smean(nmse_train_list),
        "nmse_train_std":     _sstd(nmse_train_list),
        "mc_total_mean":      _smean(mc_total_list),
        "mc_total_std":       _sstd(mc_total_list),
        "mc_per_delay_mean":  mc_per_delay_mean,
        "mc_per_delay_std":   mc_per_delay_std,
        # liste grezze per post-processing
        "S_inf_static_all":   S_static_list,
        "nmse_all":           nmse_list,
        "mc_total_all":       mc_total_list,
    }

    if compute_single_entropy:
        agg["S_single_dynamic_mean"] = _smean(S_single_list)
        agg["S_single_dynamic_std"]  = _sstd(S_single_list)
        agg["dS_single_peak_mean"]   = _smean(dS_single_list)
        agg["dS_single_peak_std"]    = _sstd(dS_single_list)
        agg["S_single_dynamic_all"]  = S_single_list
        agg["dS_single_peak_all"]    = dS_single_list

    if compute_fermionic_entropy:
        agg["S_fermionic_mean"]       = _smean(S_ferm_list)
        agg["S_fermionic_std"]        = _sstd(S_ferm_list)
        agg["dS_fermionic_peak_mean"] = _smean(dS_ferm_list)
        agg["dS_fermionic_peak_std"]  = _sstd(dS_ferm_list)
        agg["S_fermionic_all"]        = S_ferm_list
        agg["dS_fermionic_peak_all"]  = dS_ferm_list

    if save_traces:
        agg["traces"] = traces

    return agg
