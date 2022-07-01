import numpy as np


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


def opposite_eye(n: int = 3):
    a = np.eye(n)
    for c, r in enumerate(a):
        r = r[::-1]
        a[c, :] = r
    return a
