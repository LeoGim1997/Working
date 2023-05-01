import numpy as np
from math import floor
from typing import Optional, Tuple
from scipy.signal import convolve2d


def CG(s: float) -> float:
    return 1 / (np.sqrt(s))


def guassian_samplev2(l: int = 5, std: float = 1.0) -> np.ndarray:
    """Fast gaussian Sample generation
    Allows to quickly generates a gaussian x-vector

    Parameters
    ----------
    std: float

    """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(std))
    return np.reshape(gauss, (gauss.shape[0], 1))


def gaussian_sample(
    std: float = 1, min: int = -3, max: int = 3, n_samples: int = 50
) -> np.ndarray:
    """Return a 1-d Gaussian array.
    Allows for more spec for the desired output
    than `gaussian.guassian_samplev2`.

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
    x: ndarray
        1-d gaussian array.

    Notes
    -----
    The shape of the output vector is (n_samples,1)
    and not (n_samples,) for an easy reuse with numpy.dot
    to create matrix for instance.

    See Also
    --------
    gaussian_samplev2 : faster computation with no parametrization.
    """
    sqrt_pi = np.sqrt(2 * np.pi)
    x = np.linspace(min, max, n_samples)
    x = CG(std) * sqrt_pi * np.exp(-CG(std) * pow(x, 2))

    return np.reshape(x, (len(x), 1))


def gaussian_kernel(
    std: float = 1, threshold: Optional[float] = None, sf: Optional[float] = None
) -> np.ndarray:
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
    sf: Scale factor. Default to None, if specifed rescale the matrix

    Returns
    -------
        kernel: ndarray
            Gaussian kernel for convolution.
    """
    if std < 0:
        raise ValueError("The std cannot be negative")
    # haldf-witdh of the filter
    hw = floor(std) * 3
    w = 2 * hw + 1
    # w*5 : allows to add more points to the mesh grid
    x = gaussian_sample(std, -hw, hw, w * 5)
    y = x
    m = np.dot(x, y.T)

    # get only usefull values inside [-3*std,3*std]
    # if threshold is set to None (default)
    if threshold:
        c_x = w * 5 // 2
        c_y = w * 5 // 2
        cut = w * 5 // 5
        if sf:
            scale_factor = 1 / np.average(m)
            m = scale_factor * m
        kernel = m[c_x - cut : c_x + cut + 1, c_y - cut : c_y + cut + 1]
        return kernel
    else:
        x = x[x[:, 0] > threshold, :]
        y = x
        kernel = np.dot(x, y.T)
        if sf is not None:
            scale_factor = 1 / np.average(kernel)
            return scale_factor * kernel
        return kernel


def laplacianOfGaussian(
    l: int = 5, std: float = 1.0, threshold: Optional[float] = None
) -> np.ndarray:
    """Compute the Laplacian of gaussian.
    This function creates a LoG mask (matrix where values
    correpond to the laplacian of the gaussian mask.)

    Parameters
    """
    x = guassian_samplev2(l, std)
    y = x
    if threshold:
        x = x[x[:, 0] > threshold, :]
        y = x
    else:
        y = x
    mat = np.dot(x, y.T)
    # gradient convolutive kernel
    gKernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel = convolve2d(mat, gKernel, "same", "symm")
    return kernel
