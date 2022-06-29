import numpy as np
from math import floor
from im import MyImage
from typing import Iterable
import matplotlib.pyplot as plt


Matrix = Iterable[Iterable[float]]
Vector = Iterable[float]

SQRT_PI = np.sqrt(2 * np.pi)
def CG(s: float): return 1 / (np.sqrt(s))


def gaussian_sample(std: float = 1,
                    min: int = -3,
                    max: int = 3,
                    n_samples: int = 50) -> Vector:
    """Return a 1-d Gaussian array.

    Parameters
    ----------
    std: float, default 1.
        standard deviation of the gaussian distribution.
    min: int, default -3.
    max: int, default -3.
    n_samples: int default 50.
        number of samples required.

    Returns
    -------
    x: Vector
        1-d gaussian array.
    """

    x = np.linspace(min, max, n_samples)
    x = CG(std) * SQRT_PI * np.exp(-CG(std) * pow(x, 2))

    return np.reshape(x, (len(x), 1))


def gaussian_kernel(std: float = 1, threshold=None) -> Matrix:
    if std < 0:
        raise ValueError('The std cannot be negative')
    # haldf-witdh of the filter
    hw = floor(std) * 3
    w = 2 * hw + 1
    # w*5 : allows to add more points to the mesh grid
    def gauss(): return gaussian_sample(std, -hw, hw, w * 5)
    x, y = gauss(), gauss()
    m = np.dot(x, y.T)

    # get only usefull values inside [-3*std,3*std]
    # if threshold is set to None (default)
    if threshold is None:
        c_x = w * 5 // 2
        c_y = w * 5 // 2
        cut = w * 5 // 5
        kernel = m[c_x - cut:c_x + cut + 1, c_y - cut:c_y + cut + 1]
        scale_factor = 1 / np.average(kernel)
        kernel = scale_factor * kernel
        return kernel

    kernel = m[m[...] < threshold]
    scale_factor = 1 / np.average(kernel)
    kernel = scale_factor * kernel
    return kernel


a = gaussian_kernelv2(5)
b = gaussian_kernel(5)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(a)
plt.subplot(1, 2, 2)
plt.imshow(b)
plt.show()
