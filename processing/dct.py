import numpy as np
from fourrier import compute_cosinus
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg


def dct1D(input: np.ndarray) -> np.ndarray:
    '''
    Compute the discrete cosinus transform of a 1D signal
    of size N*1.
    Args:
        input (np.ndarray) : input signal of size N*1.
        dct (np.ndarray): output signal of size N*1.
    '''
    size = np.shape(input)
    if (len(size) == 2):
        if size[1] > 1:
            raise ValueError('Input signal is not 1D.')
    else:
        N = input.shape[0]
    dct = np.zeros(N)
    for k in range(N):
        cos_vec = [np.cos((2 * j + 1) * k * np.pi / 2 * N) for j in range(N)]
        cos_vec = np.array(cos_vec)
        if k == 0:
            coeff = np.sqrt(2) / N
        else:
            coeff = 1 / N
        cos_vec = coeff * cos_vec
        dct[k] = np.dot(cos_vec, input)
    return dct


def compute_dct_row(image: np.ndarray) -> np.ndarray:
    """
    Apply DCT on every row of the input image.
    Will returns the complex horizontal fourrier coefficients
    image. \\
    Args:
        image (np.array)
    Returns:
        dft_h (np.array)
    """
    N, M = np.shape(image)
    dft_h = np.zeros((N, M))
    for i in range(N):
        dft_h[i, :] = dct1D(image[i, :])
    return dft_h


def compute_dct_col(image: np.ndarray, inv=False) -> np.ndarray:
    """
    Apply DCT on every columns of the input image.
    Will returns the complex horizontal fourrier coefficients
    image. \\
    Args:
        image (np.array)
    Returns:
        dft_h (np.array)
    """
    N, M = np.shape(image)
    dft_h = np.zeros((N, M))
    for i in range(M):
        dft_h[:, i] = dct1D(image[:, i])
    return dft_h


def compute_dct_image(image: np.ndarray) -> np.ndarray:
    """
    Function to compute the 2-D DCT transform of the input
    image. \
    Args:
        image (np.array)
    Returns:
        dft_final (np.array): 2-D DFT image (complex values).
    """
    dft = compute_dct_row(image)
    dft_final = compute_dct_col(dft)
    return dft_final


img = Path(__file__).parents[1] / 'image_folder/Lena.jpeg'
data = mpimg.imread(img.as_posix())
a = compute_dct_image(data)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(a, cmap='gray')
plt.show()
