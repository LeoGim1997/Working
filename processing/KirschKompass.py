import itertools as it
from typing import Any, Callable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from im import normalize
from scipy.signal import convolve2d


def g(orientation: int = 1) -> np.ndarray:
    """Function to generate each direction
    of the Kirsch Compass.

    Parameters
    ----------
    orientation: int
        direction to compute.
    Notes
    -----
    The mapping of the direction according to z
    is the following:
    orientation = 1: N
    orientation = 2: NW
    orientation = 3: W
    orientation = 4: SW
    orientation = 5: S
    orientation = 6: SE
    orientation = 7: E
    orientation = 8: NE
    """
    g1 = np.asarray([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    g2 = np.asarray([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    mirror = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    mapDict = {
        1: g1,
        2: g2,
        3: g2.T,
        4: np.transpose(np.dot(g2, mirror)),
        5: np.transpose(np.dot(g1.T, mirror)),
        6: np.dot(np.transpose(np.dot(g2, mirror)), mirror),
        7: np.dot(g1.T, mirror),
        8: np.dot(g2, mirror),
    }
    if orientation not in mapDict:
        raise ValueError(
            f"Wrong value for Kompass input orientation={orientation}."
            f"Must be between 0 and 8."
        )
    return mapDict[orientation]


def kirschCompass(n: int = 8) -> Iterator:
    """Generates Kirsch Mask.
    This function returns a generators for
    all direction `z<=n`.

    Parameters
    ----------
    - n: int
        final rank for z

    Examples
    --------
    >>> a = kirschCompass(3)
    >>> print(type(a))
        class <GeneratorType>
    >>> a = list(a))
    >>> print(a[0])
        [[5, 5, 5],
        [-3, 0, -3],
        [-3, -3, -3]]
    """
    return (g(z) for z in range(1, n + 1))


def apply1Compass(img: np.ndarray, z: int = 1) -> np.ndarray:
    return convolve2d(img, g(z), "same", "symm")


def show_result(img: np.ndarray) -> Any:
    img = normalize(img)
    plt.figure()
    for c in range(1, 9):
        plt.subplot(2, 4, c)
        plt.imshow(apply1Compass(img, c), cmap="gray")
        plt.title(f"g({c})")
    plt.show()


def kirschEdge(image: np.ndarray) -> np.ndarray:
    """Compute Edge dectection using Kirsch Kompass.

    The value for  the output pixel `i,j` denoted `p[i,j]`
    correspond to the maximum value of `p[i,j]` among
    all convolved matrix `g(z)*image`,
    where g(z=1,2..) is a Kirsch Mask for 1 orientation.

    Parameters
    ----------
    - image: np.array
        Input image matrix.

    Returns
    -------
    - edge: np.array
        Output image matrix.

    See Also
    --------
    g: compute the Kirsch Mask for 1 direction.
    apply1Compass: returns the convolution maxtrix for 1 chosen direction.
    """
    # Brute force
    im = normalize(image)
    im = image
    all_mat = [convolve2d(im, g, "same", "symm") for g in kirschCompass()]
    edge = np.zeros((image.shape))
    n, m = im.shape
    for i, j in it.product(range(n), range(m)):
        edge[i, j] = max([all_mat[idx][i, j] for idx in range(0, 7)])
    return edge
