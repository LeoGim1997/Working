import numpy as np
import matplotlib.pyplot as plt


def G(w: float, n: float, wc=1) -> float:
    return 1 / (np.sqrt(1 + (w / wc)**2 * n))


def frequencyVector(start=0.01, stop=16, step=100) -> np.array:
    vec = np.arange(start, stop, step)
    return 2 * np.pi * vec


w_vector = frequencyVector()
response = np.array([G(w, 2, wc=1) for w in w_vector])
plt.figure()
plt.plot(np.log10(w_vector), 20 * np.log10(response))
plt.show()
