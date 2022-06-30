import numpy as np
from scipy.signal import convolve2d
from im import MyImage, normalize
import matplotlib.pyplot as plt
from types import GeneratorType


def g(z: int = 1) -> np.array:
    """Function to generate each direction
    of the Kirsch Compass.

    Parameters
    ----------
    z: int
        direction to compute.
    Notes
    -----
    The mapping of the direction according to z
    is the following:
    z = 1 : N
    z = 2 : NW
    z = 3 : W
    z = 4 : SW
    z = 5 : S
    z = 6 : SE
    z = 7 : E
    z = 8 : NE
    """
    g1 = np.asarray([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    g2 = np.asarray([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    mirror = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    if (z < 0 or z > 8) or type(z) is float:
        raise ValueError(f'Wrong value z={z}. Must be between 0 and 8.')
    if z == 1:
        return g1
    if z == 2:
        return g2
    if z == 3:
        return g1.T
    if z == 4:
        return np.transpose(np.dot(g2, mirror))
    if z == 5:
        return np.transpose(np.dot(g1.T, mirror))
    if z == 6:
        return np.dot(np.transpose(np.dot(g2, mirror)), mirror)
    if z == 7:
        return np.dot(g1.T, mirror)
    if z == 8:
        return np.dot(g2, mirror)


def kirschCompass(n: int = 8) -> GeneratorType:
    """Generates Kirsch Mask.
    This function returns a generators for
    all direction `z<=n`.

    Parameters
    ----------
    n: int
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


def apply1Compass(img: np.array, z: int = 1) -> np.array:
    return convolve2d(img, g(z), "same", "symm")


def show_result(img: MyImage) -> None:
    img = img.get_matrix(img.name)
    img = normalize(img)
    plt.figure()
    for c in range(1, 9):
        plt.subplot(2, 4, c)
        plt.imshow(apply1Compass(img, c), cmap='gray')
        plt.title(f'g({c})')
    plt.show()
