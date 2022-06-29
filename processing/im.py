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
                pixel = ((pixel + 0.055) / float(1.055))**(2.4)
    return img_1c


def gray_scale(img: np.array) -> np.array:
    norm = [0.2126, 0.7152, 0.0722]
    dim = np.shape(img)
    img2 = np.zeros((dim[0], dim[1]))
    l_matrix = [value * gray_scale_conv_1channel(img, channel) for channel, value in enumerate(norm)]
    for i in range(dim[0]):
        for j in range(dim[1]):
            p = l_matrix[0][i, j] + l_matrix[1][i, j] + l_matrix[2][i, j]
            img2[i, j] = 12.92 * p if p <= 0.0031308 else (((p + 0.055) / 1.055))**(2.4)
    img_finale = np.ones((dim[0], dim[1], dim[2] + 1))
    return img_finale


def fast_rgb2grey(img: np.array) -> np.array:
    if len(img.shape) == 3:
        return np.dot(img[..., :3], [0.299, 0.587, 0.114])
    else:
        return 0


def rotating_image(img: np.array, axis='horizontal') -> np.array:
    img_rotate = np.copy(img)
    dim = np.shape(img)
    center = int(dim[1] / 2)
    if axis == 'center':
        for j in range(dim[1]):
            img_rotate[:, j] = img[:, -j]
    if axis == 'horizontal':
        for i in range(dim[0]):
            img_rotate[i, :] = img[-i, :]
    if axis == 'diag':
        img_rotate = rotating_image(rotating_image(img, axis='center'), axis='horizontal')
    return img_rotate


# Easy image Load
class MyImage:
    def __init__(self, name='lena') -> None:
        self.name = name

    def get_matrix(self, image='lena'):
        dirImage = Path(__file__).parents[1] / 'image_folder'
        if image == 'lena':
            path = dirImage / 'lena.jpeg'
        if image == 'maison':
            path = dirImage / 'maison_alsacienne.jpeg'
            if not path.exists():
                raise FileExistsError(f'Image {path.as_posix()}')
            return mpimg.imread(path.resolve().as_posix())

    @staticmethod
    def show(img: np.array) -> None:
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.show()

    @staticmethod
    def show_compare(img1: np.array, img2: np.array, icmap='gray') -> None:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img1, cmap=icmap)
        plt.subplot(1, 2, 2)
        plt.imshow(img2, cmap=icmap)
        plt.show()
