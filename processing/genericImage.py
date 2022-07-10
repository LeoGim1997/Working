import numpy as np
from skimage.draw import line


def im1(n: int = 5) -> np.array:
    """square Matrix with upper
    assigned to 1.
    """
    a = np.ones((n, n))
    b = np.triu(a, k=0)
    c = 0 * np.tril(a, k=0)
    return b + c


def im2(n: int = 5) -> np.array:
    """square Matrix with lower
    assigned to 1.
    """
    a = np.ones((n, n))
    b = 0 * np.triu(a, k=0)
    c = np.tril(a, k=0)
    return b + c


def opposite_eye(n: int = 3) -> np.array:
    """Return the axial symettry of
    an `n*n` eye matrix.
    """
    a = np.eye(n)
    for c, r in enumerate(a):
        r = r[::-1]
        a[c, :] = r
    return a


def straight_line() -> np.array:
    """Classic straight-line Hough transform.
    Form a template matric which form an
    X with blank line.
    """
    image = np.zeros((100, 100))
    idx = np.arange(25, 75)
    image[idx[::-1], idx] = 225
    image[idx, idx] = 255
    return image


def cross_line() -> np.array:
    image = np.zeros((200, 200))
    idx = np.arange(25, 175)
    image[idx, idx] = 255
    image[line(45, 25, 25, 175)] = 255
    image[line(25, 135, 175, 155)] = 255
