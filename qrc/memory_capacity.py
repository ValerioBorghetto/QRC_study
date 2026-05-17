"""
qrc/memory_capacity.py
----------------------
Calcolo della Memory Capacity (MC) per reservoir computing.

Definizione (Jaeger 2002):
    MC_k = 1 - NMSE_k
         = corr(y(t), u(t-k))^2

dove u(t-k) è l'input ritardato di k passi e y(t) è l'output del
readout addestrato a ricostruire u(t-k).

La MC totale è:
    MC = sum_k MC_k

Riferimento:
    Jaeger, H. (2002). Short term memory in echo state networks.
    GMD Report 152, German National Research Center for Information
    Technology.
"""

import numpy as np
from tasks_fixed import train_readout, predict_readout, add_bias


def mc_single_delay(features, u_input, k, alpha_ridge=1e-4, train_frac=0.8, use_bias=True):
    """
    Calcola MC_k: capacità del reservoir di ricostruire u(t-k).

    Argomenti:
        features:     matrice (T_valid, D) delle feature del reservoir
        u_input:      segnale di input originale completo (T_total,)
        k:            delay (intero >= 1)
        alpha_ridge:  regolarizzazione ridge
        train_frac:   frazione di training
        use_bias:     aggiunge bias al readout

    Ritorna:
        mc_k: float in [0, 1]
    """
    T_valid = features.shape[0]
    T_total = len(u_input)
    T_washout = T_total - T_valid

    # Target: u(t - k) allineato con le feature (che partono da T_washout)
    # Il campione t-esimo delle feature corrisponde al tempo T_washout + t
    # quindi il target è u(T_washout + t - k) = u[T_washout - k : T_total - k]
    start = T_washout - k
    if start < 0:
        # delay maggiore del washout: non abbiamo abbastanza storia
        return 0.0

    target = u_input[start: start + T_valid].astype(float)

    if use_bias:
        X = add_bias(features)
    else:
        X = features

    T_train = int(T_valid * train_frac)
    X_train, y_train = X[:T_train], target[:T_train]
    X_test,  y_test  = X[T_train:], target[T_train:]

    if len(y_test) == 0 or np.var(y_test) < 1e-12:
        return 0.0

    W = train_readout(X_train, y_train, alpha_ridge)
    y_pred = predict_readout(X_test, W)

    # MC_k = corr^2, equivalente a 1 - NMSE solo per segnali a media zero
    # Usiamo la definizione via correlazione di Pearson (più robusta)
    if np.std(y_pred) < 1e-12:
        return 0.0

    corr = np.corrcoef(y_test, y_pred)[0, 1]
    if not np.isfinite(corr):
        return 0.0

    return float(corr ** 2)


def compute_mc(features, u_input, max_delay=20, alpha_ridge=1e-4,
               train_frac=0.8, use_bias=True):
    """
    Calcola MC_k per k = 1, ..., max_delay e la MC totale.

    Argomenti:
        features:   matrice (T_valid, D)
        u_input:    segnale di input completo (T_total,)
        max_delay:  numero massimo di delay da testare
        alpha_ridge, train_frac, use_bias: parametri readout

    Ritorna:
        dict con:
            mc_per_delay: array (max_delay,) con MC_k per k=1..max_delay
            mc_total:     somma totale
    """
    mc_per_delay = np.array([
        mc_single_delay(features, u_input, k,
                        alpha_ridge=alpha_ridge,
                        train_frac=train_frac,
                        use_bias=use_bias)
        for k in range(1, max_delay + 1)
    ])

    return {
        "mc_per_delay": mc_per_delay,
        "mc_total": float(np.sum(mc_per_delay)),
    }
