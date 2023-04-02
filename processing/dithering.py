import numpy as np
import itertools as it
from filter import image_padding


def get_new_val(old_val: float, newmapping: np.array):
    """
    Small function to map a pixel value inside
    a new range of intensity value.
    """
    idx = np.argmin(np.abs(newmapping - old_val), axis=0)
    return newmapping[idx]


def ditheringFS(img: np.array, nc) -> np.array:
    """
    Use the Floyd-Steinberg (FS) algorithm to dither an image.
    The Algorithm of FS will act as follow:
        For each pixel a new value of intensity will
        be associated with a new value inside
        the new color range.
        The error will be propagated to the following pixel.
    """
    assert len(img.shape) == 2, f"Only n*p 1D chanel image are supported for now."
    arr = image_padding(img, half_pad=1, fill_pad=False)
    width, height = img.shape
    maxitensite = np.max(img)
    newmapping = np.linspace(0, maxitensite, nc)
    for i, j in it.product(range(width), range(height)):
        old_pixel = arr[i, j].copy()
        new_pixel = get_new_val(old_pixel, newmapping)
        arr[i, j] = new_pixel
        quant_error = old_pixel - new_pixel
        # Propagation of the quantification
        # error to next pixels inside the current
        # neigborhood.
        arr[i, j + 1] += quant_error * 7 / 16
        arr[i + 1, j - 1] += quant_error * 3 / 16
        arr[i + 1, j] += quant_error * 5 / 16
        arr[i + 1, j + 1] += quant_error * 1 / 16
    return arr[1:-1, 1:-1]
