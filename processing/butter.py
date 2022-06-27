import numpy as np
import matplotlib.pyplot as plt
import math


def G(w: float, n: float, wc=1) -> float:
    return 1 / (np.sqrt(1 + math.pow(w / wc, 2 * n)))


def frequencyVector(start=0.16, stop=2e3, step=0.001, useOmaga=False) -> np.array:
    vec = np.arange(start, stop, step)
    if useOmaga:
        return vec
    return 2 * np.pi * vec


def plot_nomalized():
    start = 0
    stop = 10
    step = 0.001
    args = (start, stop, step, True)
    w_vector = frequencyVector(*args)
    plt.figure()
    legend = []
    for i in [2, 4, 8, 16]:
        response = np.array([G(w, i, wc=5) for w in w_vector])
        plt.plot(w_vector, np.abs(response))
        legend += [f'n={i}']
    plt.grid()
    plt.legend(legend)
    plt.show()
