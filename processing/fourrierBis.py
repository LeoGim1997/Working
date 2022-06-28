import numpy as np
import sys
from pathlib import Path
sys.path.append(Path(__file__).parents[1].resolve().as_posix())
from processing.fourrier import compute_DFT


def fft(x: np.array):

    if len(np.shape(x)) > 1 and isinstance(x, (np.generic, np.ndarray)):
        x = np.reshape(x, (x.shape[0],))
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError(f"wrong shape {N} for x. Must be a power of 2.")
    elif N <= 2:
        return compute_DFT(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N / 2)] * X_odd,
                               X_even + terms[int(N / 2):] * X_odd])


def fft_v(x,):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError(f"incorrect shape N={N}. Must be a power of 2")

    N_min = min(N, 2)

    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
                       / X.shape[0])[:, None]
        X = np.vstack([X_even + terms * X_odd,
                       X_even - terms * X_odd])
    return X.ravel()
