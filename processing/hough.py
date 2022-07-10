import numpy as np
from typing import Tuple
from im import MyImage


def hough_line(img: np.array, theta=None) -> Tuple:
    """Compute the Hough line detector.
    This function generate the Hough Parameters Space
    for line detection inside a thresholded image.

    Parameters
    ----------

    img: np.array
        Input NxM image matrix.
    theta: np.array
        theta array for the angle bound.

    Returns
    -------
    accum: np.array
        Accumulator array for the paramters space (`rho`,`theta`).
    theta: np.array
        The input theta array of the defaulted value.
    bins: np.array
        the number of possible rho value for the input.

    Notes
    -----
    As a line is by hypthothesis belonging to the edges of the
    image, a preprocessing of the input image need to be done.
    This function does not do it. In order to this function
    to work correctly apply an edge detector + threshold to the
    image.

    The theta vector represent all possibles angle value for a
    given line crossing a point. If theta is not specified
    this function will define a theta vector with values in \ 
    `[-np.pi/2,np.pi/2]`.


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
