import numpy as np
import itertools as it
from .filter import image_padding
from typing import Iterable, Tuple
from enum import Enum


class Direction(Enum):
    POSITIVE = 1
    NEGATIVE = -1


class Dithering(Enum):
    """
    Enum class to represent
    the different type of dithering.
    """

    FSdithering = "FS"  # Floyd-Steinberg
    JJN = "JJN"  # Jarvis, Judice, Ninke


def get_new_val(old_val: float, newmapping: np.array) -> float:
    """
    Small function to map a pixel value inside
    a new range of intensity value.
    """
    idx = np.argmin(np.abs(newmapping - old_val), axis=0)
    return newmapping[idx]


def get_new_val_th(old_val: float, newmapping: np.array) -> float:
    """
    New way to get the new intensity value from a pixel with a
    threshold effect.
    """
    pass


def snakePath(n: int, m: int) -> Iterable[Tuple[int, int]]:
    """
    Small function to compute the iterator to go
    through the matrix.
    """
    for i in range(n):
        if i % 2 == 0:
            col = range(0, m)
        else:
            col = range(-1, -m - 1, -1)
        for j in col:
            yield (i, j)


def generatePath(case: str, n: int, m: int) -> Iterable:
    match case:
        case "normal":
            return it.product(range(n), range(m))
        case "snake":
            return snakePath(n, m)


def get_new_val_selector(case: str, old_val: float, newmapping: np.array) -> int:
    match case:
        case "normal":
            return get_new_val(old_val, newmapping)
        case "threshold":
            return get_new_val_th(old_val, newmapping)


def update_matrix(
    dithering: Dithering,
    direction: Direction,
    arr: np.array,
    quant_error: float,
    i: int,
    j: int,
) -> None:
    """
    Update the matrix for multiple direction
    """
    match dithering:
        case Dithering.FSdithering:
            if direction == Direction.POSITIVE:
                arr[i, j + 1] += quant_error * 7 / 16
                arr[i + 1, j - 1] += quant_error * 3 / 16
                arr[i + 1, j] += quant_error * 5 / 16
                arr[i + 1, j + 1] += quant_error * 1 / 16
            if direction == Direction.NEGATIVE:
                arr[i, j - 1] += quant_error * 7 / 16
                arr[i + 1, j + 1] += quant_error * 3 / 16
                arr[i + 1, j] += quant_error * 5 / 16
                arr[i + 1, j - 1] += quant_error * 1 / 16
        case Dithering.JJN:
            arr[i, j + 1] += quant_error * 7 / 48
            arr[i, j + 2] += quant_error * 5 / 48
            arr[i + 1, j - 2] += quant_error * 3 / 48
            arr[i + 1, j - 1] += quant_error * 5 / 48
            arr[i + 1, j] += quant_error * 3 / 48
            arr[i + 2, j - 2] += quant_error * 1 / 48
            arr[i + 2, j - 1] += quant_error * 3 / 48
            arr[i + 2, j] += quant_error * 5 / 48
            arr[i + 2, j + 1] += quant_error * 3 / 48
            arr[i + 2, j + 2] += quant_error * 1 / 48


def dithering(
    img: np.array,
    nc,
    case="normal",
    newvaltype="normal",
    dithering: str = "FS",
) -> np.array:
    """
    Main method for dithering algorithms to dither an image.
    2 method are implemented for now:
     - Floyd-Steinberg (FS)
     - Jarvis, Judice, Ninke (JJN) \\
    All the algorithms use the error diffusion method.
    Each process pixel will be quantifiy on a other grayscale
    encoded by the parameter `number of colors` nc.

    Parameters
    ----------
    img:np.array
        A N*P input matrix image.
    nc: int
        Number of color inside the final image.
    case: Optinnal str
        Can take 2 value : normal of snake
        Default to normal, determine the path following
        for the pixel processing. If set to snake the path
        following will go from left to right and left to right.
    newvaltype: Optionnal str
        for the new pixel selection.
    dithering: str
        Default to FS. Select the type of dithering to apply
        to the input image.
    Returns
    -------
    img: np.array
        output dithered image.
    """
    assert len(img.shape) == 2, f"Only n*p 1D chanel image are supported for now."
    half_pad = 1 if dithering == Dithering.FSdithering else 2
    arr = image_padding(img, half_pad=half_pad, fill_pad=False)
    width, height = img.shape
    maxitensite = np.max(img)
    newmapping = np.linspace(0, maxitensite, nc)
    # Classical order for matrix exploration
    # to it.product here will generate all the index of the
    # pixels starting from the top-left of the image to the
    # bottom right of the image.

    for i, j in generatePath(case, width, height):
        old_pixel = arr[i, j].copy()
        new_pixel = get_new_val_selector(newvaltype, old_pixel, newmapping)
        arr[i, j] = new_pixel
        quant_error = old_pixel - new_pixel
        # Propagation of the quantification
        # error to next pixels inside the current
        # neigborhood.
        if case == "normal":
            direction = Direction.POSITIVE
        if case == "snake":
            if j < 0:
                direction = Direction.NEGATIVE
            else:
                direction = Direction.POSITIVE
        match dithering:
            case "FS":
                dtype = Dithering.FSdithering
            case "JJN":
                dtype = Dithering.JJN
            case _:
                raise ValueError(
                    f"Incorrect type of dithering entered : {dithering}",
                    f"valid values are FS and JJN.",
                )
        update_matrix(dtype, direction, arr, quant_error, i, j)
    return arr[1:-1, 1:-2]
