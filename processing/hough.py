import numpy as np
from typing import Tuple
from im import MyImage


def hough_line(img: np.array, theta=None) -> Tuple:
    """
    Compute the hough line detector inside for the
    edges of the images.
    """
    if theta is None:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)

    # Accumulator creation (Hough space)
    offset = np.ceil(np.sqrt(img.shape[0]**2 + img.shape[1]**2))
    max_distance = int(2 * offset) + 1
    accum = np.zeros((max_distance, theta.shape[0]))
    bins = np.linspace(-offset, offset, max_distance)

    y_indx, x_indx = np.nonzero(img)
    ndixs = y_indx.shape[0]
    ntheta = theta.shape[0]

    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    for i in range(ndixs):
        x, y = x_indx[i], y_indx[i]
        for j in range(ntheta):
            accum_idx = np.round(ctheta[j] * x + stheta[j] * y) + offset
            accum[int(accum_idx), j] += 1
    return accum, theta, bins


def show_hough_space(img: np.array) -> None:
    h, theta, bins = hough_line(img)
    MyImage.show_compare(img, np.log(1 + h))
