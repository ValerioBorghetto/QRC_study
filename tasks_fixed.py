import numpy as np


def generate_narma10(T, seed=42):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=T)

    y = np.zeros(T)

    for t in range(10, T):
        y[t] = (
            0.3 * y[t - 1]
            + 0.05 * y[t - 1] * np.sum(y[t - 10:t])
            + 1.5 * u[t - 10] * u[t - 1]
            + 0.1
        )

    return u, y


def add_bias(features):
    return np.hstack(
        [
            features,
            np.ones((features.shape[0], 1)),
        ]
    )


def train_readout(features_train, target_train, alpha_ridge=1e-4):
    X = features_train
    y = target_train

    A = X.T @ X + alpha_ridge * np.eye(X.shape[1])
    b = X.T @ y

    W = np.linalg.solve(A, b)
    return W


def predict_readout(features, W):
    return features @ W


def nmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true)

    if var < 1e-12:
        return np.inf

    return mse / var


def evaluate_reservoir(
    features,
    target,
    train_frac=0.8,
    alpha_ridge=1e-4,
    use_bias=True,
):
    if use_bias:
        features = add_bias(features)

    T = len(target)
    T_train = int(T * train_frac)

    X_train = features[:T_train]
    y_train = target[:T_train]

    X_test = features[T_train:]
    y_test = target[T_train:]

    W = train_readout(X_train, y_train, alpha_ridge)

    y_pred_train = predict_readout(X_train, W)
    y_pred_test = predict_readout(X_test, W)

    return {
        "nmse_train": nmse(y_train, y_pred_train),
        "nmse_test": nmse(y_test, y_pred_test),
        "W": W,
        "y_pred_test": y_pred_test,
        "y_true_test": y_test,
        "T_train": T_train,
    }


def safe_stderr(x):
    """
    Standard error della media (std / sqrt(n)).
    Restituisce nan se x è None, vuoto, o ha un solo elemento.

    Arguments:
        x: lista o array di valori

    Returns:
        Standard error come float, oppure nan
    """
    if x is None or len(x) == 0:
        return np.nan
    if len(x) == 1:
        return np.nan
    return float(np.std(x) / np.sqrt(len(x)))


def is_valid_timeseries(u, y):
    """
    Controlla che la serie NARMA-10 sia numericamente valida:
    - nessun NaN o Inf in u o y
    - y non costante (varianza non nulla), altrimenti la NMSE è indefinita

    Arguments:
        u: segnale di input (numpy array)
        y: serie target (numpy array)

    Returns:
        True se la serie è utilizzabile, False altrimenti
    """
    if not np.all(np.isfinite(u)):
        return False
    if not np.all(np.isfinite(y)):
        return False
    if np.var(y) < 1e-12:
        return False
    return True


def is_finite_scalar(x):
    """
    Controlla che x sia uno scalare finito (non NaN, non Inf).

    Arguments:
        x: valore scalare

    Returns:
        True se finito, False altrimenti
    """
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False