import numpy as np
from math import floor
from im import MyImage
from typing import Iterable

Matrix = Iterable[Iterable[float]]
Vector = Iterable[float]

SQRT_PI = np.sqrt(2 * np.pi)
def CG(s): return 1 / (SQRT_PI * s)


def gaussian_sample(std: float = 1,
                    min: int = -3,
                    max: int = 3,
                    n_samples: int = 50) -> Vector:
    '''
    Return a 1-d Gaussian array of std sigma
    '''
    coeff = 1 / np.sqrt(std**2)
    coeff_pi = coeff / np.sqrt(2 * np.pi)

    x = np.linspace(min, max, n_samples)
    x = coeff_pi * np.exp(-coeff * pow(x, 2))

    return np.reshape(x, (len(x), 1))


def gaussian_kernel(sigma: float = 1) -> np.array:
    '''
    Return normalized gaussian kernel of std sigma
    of half-width h=3*sigma
    '''
    if sigma < 0:
        raise ValueError('The std cannot be negative')
    half_w = floor(sigma) * 3
    witdh = 2 * half_w + 1
    # add more point to the mesh-grid
    x = gaussian_sample(sigma, -half_w, half_w, witdh * 5)
    shape_x = np.shape(x)[0]
    y = np.copy(x)
    gaussian_mat = np.dot(x, np.transpose(x))
    scale_factor = 1 / np.average(gaussian_mat)
    gaussian_mat = scale_factor * gaussian_mat
    # cropping the image to only get usefull value
    n, m = np.shape(gaussian_mat)
    center_x = witdh * 5 // 2
    center_y = witdh * 5 // 2
    cut = witdh * 5 // 5
    kernel = gaussian_mat[center_x - cut - 1:center_x + cut, center_y - cut - 1:center_y + cut]
    scale_factor = 1 / np.average(kernel)
    kernel = scale_factor * kernel
    return kernel


def gaussian_kernelv2(std: float = 1, threshold=None) -> Matrix:
    if std < 0:
        raise ValueError('The std cannot be negative')
    # haldf-witdh of the filter
    h_w = floor(std) * 3
    w = 2 * h_w + 1
    # w*5 : allows to add more points to the mesh grid
    def gauss(): return gaussian_sample(std, -w, w, w * 5)
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
MyImage.show(a)
