import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def compute_gradient(img: np.array,
                     operator: str = 'Sobel',
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
        Operator to compute the gradient (Default is Sobel).
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
    if operator == 'Sobel':
        a1 = np.matrix([1, 2, 1])
        a2 = np.matrix([-1, 0, 1])
        Kx = a1.T * a2
        Ky = a2.T * a1
    if operator == 'Robert':
        Kx = np.array([[1, 0], [0, -1]])
        Ky = np.array([[0, 1], [-1, 0]])
    # Apply the Sobel operator
    Gx = convolve2d(img, Kx, "same", "symm")
    Gy = convolve2d(img, Ky, "same", "symm")
    G = np.abs(Gx) + np.abs(Gy)
    if return_xy_gradient:
        return G, Gx, Gy
    return G


def analyze_gradient(img: np.array,
                     operator: str = 'Sobel',
                     ) -> None:
    G, Gx, Gy = compute_gradient(img=img,
                                 operator=operator,
                                 return_xy_gradient=True)
    x, y = np.ravel(Gx), np.ravel(Gy)
    plt.figure()
    plt.scatter(x, y)
    plt.title('Scatter plot of gradient points.')
    plt.xlabel('\033[1m' + 'Ix' + '\033[0m')
    plt.xlabel('\033[1m' + 'Iy' + '\033[0m')
    plt.show()


from im import MyImage
a = MyImage('lena').get_matrix('chessboard')
analyze_gradient(a)
