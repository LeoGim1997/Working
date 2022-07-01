import numpy as np
from gradient_filtering import compute_gradient
from im import MyImage
from genericImage import opposite_eye
from filter import image_padding
import matplotlib.pyplot as plt
from typing import Tuple
from im import MyImage, normalize

# 1 : Take the grayscale of the image
# 2 : Apply gaussian fitler to smooth noise (Optional)
# 3 : Apply Sobel operator to find x and y gradient
# 4 : For each pixel, consider a 3*3 window around it
# and compute the corner strength function
# 5 : Find all pixels that exceed a certain threshold
#  and are the local maxima within
#   a certain window
#   (to prevent redundant dupes of features).

a = MyImage().get_matrix('chessboard')
g, gx, gy = compute_gradient(a, operator='Sobel', return_xy_gradient=True)


def haarisResponse(k: float = 0.004, num: int = 100,
                   revert: bool = False) -> np.array:

    l1 = np.linspace(0, 1, num=num)
    e = opposite_eye(num)
    xl1, xl2 = np.meshgrid(l1, l1)
    r = np.zeros((l1.shape[0], l1.shape[0]))
    for i in range(0, l1.shape[0]):
        for j in range(0, l1.shape[0]):
            f = xl1[i, j] * xl2[i, j] - k * (xl1[i, j] + xl2[i, j])**2
            r[i, j] = f
    if revert:
        return np.transpose(np.dot(r, opposite_eye(500)))
    return r


def showGlobal_Response() -> None:
    plt.figure()
    plt.imshow(haarisResponse(num=500, revert=False),
               cmap='gray',
               origin='lower',
               extent=(0, 1, 0, 1))
    plt.xlabel('\u03BB1')
    plt.ylabel('\u03BB2')
    plt.title('Response for Haaris corner detector.')
    plt.show()


def ellipse_Response(Ix: np.array, Iy: np.array,
                     k: float = 0.004) -> float:

    a = np.sum(np.dot(Ix, Ix))
    b = 2 * np.sum(np.dot(Ix, Iy))
    c = np.sum(np.dot(Iy, Iy))
    l1 = 1 / 2 * (a + c + np.sqrt(b**2 + (a - c)**2))
    l2 = 1 / 2 * (a + c - np.sqrt(b**2 + (a - c)**2))
    return l1 * l2 - k * (l1 + l2)**2


def tensor_Response(Ix: np.array, Iy: np.array,
                    k: float = 0.04) -> float:
    # TODO : Fix the tensor
    a = np.average(np.dot(Ix, Ix))
    b = 2 * np.average(np.dot(Ix, Iy))
    c = np.average(np.dot(Iy, Iy))
    M = np.asarray([[a, b], [b, c]])

    return np.linalg.det(M) - k * np.sum(np.diag(M))**2


def window_Response(Ix: np.array, Iy: np.array,
                    k: float = 0.04, method='ellipse') -> float:
    mapDict = {
        'ellipse': ellipse_Response,
        'tensor': tensor_Response
    }
    args = (Ix, Iy, k)
    if mapDict.get(method) is None:
        raise ValueError(f'method {method} does not exist.')
    return mapDict[method](*args)


def matrixResponse(img: np.array, k: float = 0.04,
                   hp: int = 1, method='ellipse') -> np.array:
    _, Gx, Gy = compute_gradient(img, operator='Sobel', return_xy_gradient=True)
    Gx, Gy = Gx - np.mean(Gx), Gy - np.mean(Gy)
    Gx, Gy = image_padding(Gx, half_pad=hp), image_padding(Gy, half_pad=hp)
    n, m = Gx.shape
    mat = np.copy(Gx)
    for i in range(hp, n - hp):
        for j in range(hp, m - hp):
            Ix = Gx[i - hp:i + hp + 1, j - hp:j + hp + 1]
            Iy = Gy[i - hp:i + hp + 1, j - hp:j + hp + 1]
            mat[i, j] = window_Response(Ix, Iy, k=k, method=method)
    return mat[hp:-hp, hp:-hp]


def non_MaximalSupression(img: np.array, hp: int = 1) -> np.array:
    img = image_padding(img=img, half_pad=hp)
    f = np.copy(img)
    n, m = img.shape
    for i in range(hp, n - hp):
        for j in range(hp, m - hp):
            submat = img[i - hp:i + hp + 1, j - hp:j + hp + 1]
            if (img[i, j] == np.max(submat)):
                f[i, j] = img[i, j] * 5
            else:
                f[i, j] = 0
    return f


a = MyImage('lena').get_matrix(fullpath='/Users/leogimenez/Desktop/git_depo_local/Working/image/image_folder/bbc-logo.jpeg')
a = a[..., 0]
a = normalize(a)
b = matrixResponse(a, k=0.04, method='ellipse')
MyImage.show(b, icmap='magma')
