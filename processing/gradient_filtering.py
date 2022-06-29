import numpy as np
from scipy.signal import convolve2d


def compute_gradient(img: np.array, operator='Sobel', return_xy_gradient: bool = False) -> np.array:
    '''
    Function to compute the gradient of the image according
    to certain kernel operator (default is Sobel operator)    
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
