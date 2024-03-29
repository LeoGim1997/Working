import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from pathlib import Path

# Basic image processing


def remove_channels(img: np.array, channels: int) -> np.array:
    img_c = np.copy(img)
    shape = np.shape(img)
    width = shape[0]
    height = shape[1]
    for i in range(width):
        for j in range(height):
            img_c[i][j][channels] = 0
    return img_c


def addition_component(img: np.array):
    img_c = np.copy(img)
    shape = np.shape(img)
    width = shape[0]
    height = shape[1]
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            img_c[i][j] = img_c[i][j] + img_c[i - 1][j - 1]
    return img_c


def normalize(img: np.array) -> np.array:
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def integral_image_recursive_1channel(img: np.array) -> np.array:
    shape = np.shape(img)
    width = shape[0]
    height = shape[1]
    I = np.zeros((width, height))
    # Init
    I[0, 0] = img[0, 0]
    for i in range(1, width):
        for j in range(1, height):
            I[i, j] = img[i, j] + I[i, j - 1] + I[i - 1, j] - I[i - 1, j - 1]
    return I


def gray_scale_conv_1channel(img: np.array, channel=0) -> np.array:
    img_1c = normalize(img[:, :, channel])
    for row in img_1c:
        for pixel in row:
            if pixel <= 0.04045:
                pixel = pixel / float(12.92)
            if pixel > 0.04045:
                pixel = ((pixel + 0.055) / float(1.055)) ** (2.4)
    return img_1c


def gray_scale(img: np.array) -> np.array:
    norm = [0.2126, 0.7152, 0.0722]
    dim = np.shape(img)
    img2 = np.zeros((dim[0], dim[1]))
    l_matrix = [
        value * gray_scale_conv_1channel(img, channel)
        for channel, value in enumerate(norm)
    ]
    for i in range(dim[0]):
        for j in range(dim[1]):
            p = l_matrix[0][i, j] + l_matrix[1][i, j] + l_matrix[2][i, j]
            img2[i, j] = (
                12.92 * p if p <= 0.0031308 else (((p + 0.055) / 1.055)) ** (2.4)
            )
    img_finale = np.ones((dim[0], dim[1], dim[2] + 1))
    return img_finale


def fast_rgb2grey(img: np.array) -> np.array:
    img = np.asarray(img)

    if len(img.shape) != 3:
        raise ValueError(f"Wrong dim for input image. shape={img.shape}")
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def rotating_image(img: np.array, axis="horizontal") -> np.array:
    img_rotate = np.copy(img)
    dim = np.shape(img)
    center = int(dim[1] / 2)
    if axis == "center":
        for j in range(dim[1]):
            img_rotate[:, j] = img[:, -j]
    if axis == "horizontal":
        for i in range(dim[0]):
            img_rotate[i, :] = img[-i, :]
    if axis == "diag":
        img_rotate = rotating_image(
            rotating_image(img, axis="center"), axis="horizontal"
        )
    return img_rotate


def imageThreshold(img: np.array, th: float) -> np.array:
    """Threshold the input image
    this function take the input threshold to binarize
    the image and returns a copy

    Parameters
    ----------

    img: np.array
        The input image as an array.
    th: np.array
        the selected threshold.

    Returns
    -------

    img: np.array
        the thresholded image.

    Notes:
    The value below `th` will be considered as `0`.
    """
    a = np.copy(img)
    a[a >= th] = 1
    a[a < 1] = 0
    return a


# Easy image Load


class MyImage:
    def __init__(self, name="lena") -> None:
        self.name = name
        self._dirImage  = Path(__file__).parents[1] / "image_folder"

    def get_matrix(self, fullpath: str = None):
        if fullpath is not None:
            return mpimg.imread(fullpath)

        mapDict = {
            "lena": self._dirImage / "lena.jpeg",
            "house": self._dirImage / "maison_alsacienne.jpeg",
            "temple": self._dirImage / "Boxfilter_pavilion_original.jpg",
            "chessboard": self._dirImage / "chessboard_GRAY.png",
            "bbc": self._dirImage / "bbc-logo.jpeg",
            "david": self._dirImage / f"david.png",
        }
        path = mapDict.get(self.name)
        if path is None:
            raise FileExistsError(f"Image {path.as_posix()} do not exists.")
        return mpimg.imread(path.resolve().as_posix())

    @staticmethod
    def show(img: np.array, icmap="gray") -> None:
        plt.figure()
        plt.imshow(img, cmap=icmap)
        plt.show()

    @staticmethod
    def show_compare(img1: np.array, img2: np.array, icmap="gray") -> None:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img1, cmap=icmap)
        plt.subplot(1, 2, 2)
        plt.imshow(img2, cmap=icmap)
        plt.show()
