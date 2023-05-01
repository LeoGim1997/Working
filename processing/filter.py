from __future__ import annotations

import itertools as it

import numpy as np
from gaussian import gaussian_kernel
from scipy.signal import convolve2d


def gaussian_blur(img: np.ndarray, sigma: float = 1) -> np.ndarray:
    """
    Apply a 7x7 gaussian blur on a img (default)
    on 1 channel of the input image
    """
    filter = gaussian_kernel(std=sigma)
    return convolve2d(img, filter, "same", "symm")


def image_padding(
    img: np.ndarray, half_pad: int = 4, fill_pad=False, defaultfill=1
) -> np.ndarray:
    """
    Function returning a padded image use for convolution
    with a square fitler of half_with = half_pad.
    The center of the image will be the orginal image
    """
    shape = np.shape(img)
    if not len(shape) >= 2:
        raise ValueError(f"Dimension of the input matrix must be at least 2.")
    n, m = shape[0], shape[1]
    hw = half_pad
    # pad the image for fitting
    if defaultfill == 1:
        img_c = np.ones((n + 2 * hw, m + 2 * hw))
        img_c[hw:-hw, hw:-hw] = img
    else:
        img_c = defaultfill * np.ones((n + 2 * hw, m + 2 * hw))
        img_c[hw:-hw, hw:-hw] = img

    # filling the 4 corner with the same corner value as the original image
    if fill_pad and half_pad != 1:
        # top block of the matrix
        img_c[0:hw, 0:hw] = img[0, 0] * img_c[0:hw, 0:hw]
        for i in range(hw):
            img_c[i, hw:-hw] = img[0, :]
        img_c[0:hw, -hw:] = img[0, -1] * img_c[0:hw, -hw:]
        # bottom block of the matrix
        img_c[-hw:, 0:hw] = img[-1, 0] * img_c[-hw:, 0:hw]
        for i in range(1, hw):
            img_c[-i, hw:-hw] = img[-i, :]
        img_c[-hw:, -hw:] = img[-1, -1] * img_c[-hw:, -hw:]

        # left and right block of the matrix
        img_c[hw:-hw, :hw] = img[:, :hw]
        img_c[hw:-hw, -hw:] = img[:, -hw:]
        return img_c

    if fill_pad and half_pad == 1:
        img_c[0:hw, 0:hw] = img[0, 0] * img_c[0:hw, 0:hw]
        for i in range(hw):
            img_c[i, hw:-hw] = img[0, :]
        img_c[0:hw, -hw:] = img[0, -1] * img_c[0:hw, -hw:]
        # bottom block of the matrix
        img_c[-hw:, 0:hw] = img[-1, 0] * img_c[-hw:, 0:hw]

        img_c[-1, 1:-1] = img[-1, :]
        img_c[-hw:, -hw:] = img[-1, -1] * img_c[-hw:, -hw:]

        # left and right block of the matrix
        img_c[hw:-hw, :hw] = img[:, :hw]
        img_c[hw:-hw, -hw:] = img[:, -hw:]
        return img_c
    return img_c


def sobel_filter_horizontal(img: np.ndarray) -> np.ndarray:
    """Compute the horizontal image gradient
    using Sobel convolution kernel.

    Parameters
    ----------
    img (np.ndarray) : Input array

    Returns
    -------
    imgSobel (np.array) : Convoluted image.
    """

    filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Gx = image_padding(img, 1, fill_pad=False)
    img = np.zeros(np.shape(Gx))
    N, M = np.shape(Gx)
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            sub_matrix = Gx[i - 1 : i + 2, j - 1 : j + 2]
            prod = np.multiply(sub_matrix, filter)
            img[i, j] = np.sum(prod)
    return img[1:-1, 1:-1]


def sobel_filter_vertical(img: np.ndarray) -> np.ndarray:
    """Compute thevertical image gradient
        using Sobel convolution kernel.

    Parameters
    ----------
    img (np.ndarray) : Input array

    Returns
    -------
    imgSobel (np.array) : Convoluted image.
    """
    filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    Gy = image_padding(img, 1, fill_pad=False)
    img = np.zeros(np.shape(Gy))
    N, M = np.shape(Gy)
    for i, j in it.product(range(1, N - 1), range(1, M - 1)):
        sub_matrix = Gy[i - 1 : i + 2, j - 1 : j + 2]
        img[i, j] = np.sum(np.multiply(sub_matrix, filter))
    return img[1:-1, 1:-1]


def normalized_sobel_filter(img: np.ndarray, threshold: int = 0) -> np.ndarray:
    """Compute the normalized sobel filter on both
    horizontal. Can be use for edge detection inside an image.

    Parameters
    ----------
    img (np.ndarray) : Input array
    threshold (int) : Theshold to apply to the gradient matrix
                      norm. (Nb: Every value below are 0-assigned.)

    Returns
    -------
    imgSobelNormalized (np.array) : Convoluted image.
    """
    Gx = sobel_filter_horizontal(img)
    Gy = sobel_filter_vertical(img)
    n, m = img.shape
    if len(img.shape) != 2:
        # TODO : Take into account the case of multi channel image.
        raise ValueError(f"Only 1D matrix are availaible for now.")
    img_f = np.abs(Gx) + np.abs(Gy)
    if threshold != 0:
        for i, j in it.product(range(n), range(m)):
            if img_f[i, j] < threshold:
                img_f[i, j] = 0
            else:
                img_f[i, j] = 255
    return img_f
