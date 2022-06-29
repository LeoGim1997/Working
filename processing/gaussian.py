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

    Notes
    -----
    The shape of the output vector is (n_samples,1) 
    and not (n_samples,) for an easy reuse with numpy.dot
    to create matrix for instance.

    """

    x = np.linspace(min, max, n_samples)
    x = CG(std) * SQRT_PI * np.exp(-CG(std) * pow(x, 2))

    return np.reshape(x, (len(x), 1))


def gaussian_kernel(std: float = 1, threshold: float = None, sf=None) -> Matrix:
    """Generate a gaussian Kernel.
    This function returns a matrix corresponding to a
    a gaussian kernel with parameters `mu=0 and sigma=`std`.
    By Default, the witdh of the filter will be
    `w=3*std`

    Parameters
    ----------
    std: float, default 1.
        standard deviation fo the gaussian function.
    threshold: float, default None.

    Returns
    -------
        kernel: Matrix
            Gaussian kernel for convolution.
    """
    if std < 0:
        raise ValueError('The std cannot be negative')
    # haldf-witdh of the filter
    hw = floor(std) * 3
    w = 2 * hw + 1
    # w*5 : allows to add more points to the mesh grid
    x = gaussian_sample(std, -hw, hw, w * 5)
    y = x
    m = np.dot(x, y.T)

    # get only usefull values inside [-3*std,3*std]
    # if threshold is set to None (default)
    if threshold is None:
        c_x = w * 5 // 2
        c_y = w * 5 // 2
        cut = w * 5 // 5
        if sf is not None:
            scale_factor = 1 / np.average(m)
            m = scale_factor * m
        kernel = m[c_x - cut:c_x + cut + 1, c_y - cut:c_y + cut + 1]
        return kernel
    else:
        x = x[x[:, 0] > threshold, :]
        y = x
        kernel = np.dot(x, y.T)
        if sf is not None:
            return scale_factor * kernel
        return kernel
