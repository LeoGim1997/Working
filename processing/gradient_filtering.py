import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def sobel():
    a1 = np.matrix([1, 2, 1])
    a2 = np.matrix([-1, 0, 1])
    Kx = a1.T * a2
    Ky = a2.T * a1
    return Kx, Ky


def robert():
    Kx = np.array([[1, 0], [0, -1]])
    Ky = np.array([[0, 1], [-1, 0]])
    return Kx, Ky


def canny():
    Kx = np.array([-1, 0, 1])
    Ky = np.array([1, 0, 1])
    Kx = np.reshape(Kx, (1, 3))
    Ky = np.reshape(Kx, (3, 1))
    return Kx, Ky.T


def compute_gradient(img: np.array,
                     operator: str = 'sobel',
                     return_xy_gradient: bool = False) -> np.array:
    '''Compute gradient.
    This function compute the gradient image (G)
    of the input image `img`.
    by convolution with an operator.

    Parameters:
    ----------
    img: np.array
        Input image.
    operator: str
        Operator to compute the gradient (Default is Sobel operator).
        Possible values can be:

        - sobel
        - canny
        - robert

    return_xy_gradient: bool
        Allows to returns Gx,Gy in additon to G

    Returns
    -------
    G: np.array
        Gradient image.
    G,Gx,Gy: tuple(np.array)
        If `return_xy_gradient=True`,
        returns gradient matrix in x,y direction.
    '''

    mapDict = {
        'sobel': sobel,
        'robert': robert,
        'canny': canny
    }

    func = mapDict.get(operator)
    if func is None:
        raise ValueError(f'No operator called {operator} found.')
    Kx, Ky = func()
    # Apply the selected operator
    Gx = convolve(img, Kx)
    Gy = convolve(img, Ky)
    G = np.abs(Gx) + np.abs(Gy)
    if return_xy_gradient:
        return G, Gx, Gy
    return G


def distribution(img: np.array,
                 operator: str = 'Sobel',
                 ) -> None:
    G, Gx, Gy = compute_gradient(img=img,
                                 operator=operator,
                                 return_xy_gradient=True)
    x, y = np.ravel(Gx), np.ravel(Gy)
    mx, my = np.max(x), np.min(y)
    plt.figure()
    plt.scatter(x, y)
    plt.xlim((-2 * mx, 2 * mx))
    plt.xlim((-2 * my, 2 * my))
    plt.title('Distribution of image gradients.')
    plt.xlabel('Ix')
    plt.ylabel('Iy')
    plt.show()
