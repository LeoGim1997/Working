from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from filter import image_padding
from genericImage import opposite_eye
from gradient_filtering import compute_gradient


def haarisResponse(
    k: float = 0.004, num: int = 100, revert: bool = False
) -> np.ndarray:
    l1 = np.linspace(0, 1, num=num)
    e = opposite_eye(num)
    xl1, xl2 = np.meshgrid(l1, l1)
    r = np.zeros((l1.shape[0], l1.shape[0]))
    for i in range(0, l1.shape[0]):
        for j in range(0, l1.shape[0]):
            f = xl1[i, j] * xl2[i, j] - k * (xl1[i, j] + xl2[i, j]) ** 2
            r[i, j] = f
    if revert:
        return np.transpose(np.dot(r, opposite_eye(500)))
    return r


def showGlobal_Response() -> Any:
    plt.figure()
    plt.imshow(
        haarisResponse(num=500, revert=False),
        cmap="gray",
        origin="lower",
        extent=(0, 1, 0, 1),
    )
    plt.xlabel("\u03BB1")
    plt.ylabel("\u03BB2")
    plt.title("Response for Haaris corner detector.")
    plt.show()


def ellipse_Response(Ix: np.ndarray, Iy: np.ndarray, k: float = 0.004) -> float:
    a = np.sum(np.multiply(Ix, Ix))
    b = 2 * np.sum(np.multiply(Ix, Iy))
    c = np.sum(np.multiply(Iy, Iy))
    l1 = 1 / 2 * (a + c + np.sqrt(b**2 + (a - c) ** 2))
    l2 = 1 / 2 * (a + c - np.sqrt(b**2 + (a - c) ** 2))
    r = l1 * l2 - k * (l1 + l2) ** 2
    return r


def tensor_Response(Ix: np.ndarray, Iy: np.ndarray, k: float = 0.04) -> float:
    # TODO : Fix the tensor
    a = np.sum(np.multiply(Ix, Ix))
    b = 2 * np.sum(np.multiply(Ix, Iy))
    c = np.sum(np.multiply(Iy, Iy))
    M = np.asarray([[a, b], [b, c]])

    return np.linalg.det(M) - k * np.sum(np.diag(M)) ** 2


def window_Response(
    Ix: np.ndarray, Iy: np.ndarray, k: float = 0.04, method="ellipse"
) -> float:
    mapDict = {"ellipse": ellipse_Response, "tensor": tensor_Response}
    args = (Ix, Iy, k)
    if mapDict.get(method) is None:
        raise ValueError(f"method {method} does not exist.")
    return mapDict[method](*args)


def matrixResponse(
    img: np.ndarray, k: float = 0.04, hp: int = 1, method: str = "ellipse"
) -> np.ndarray:
    _, Gx, Gy = compute_gradient(img, operator="Sobel", return_xy_gradient=True)
    Gx, Gy = image_padding(Gx, half_pad=hp), image_padding(Gy, half_pad=hp)
    n, m = Gx.shape
    mat = np.zeros((n, m))
    for i in range(hp, n - hp):
        for j in range(hp, m - hp):
            Ix = Gx[i - hp : i + hp + 1, j - hp : j + hp + 1]
            Iy = Gy[i - hp : i + hp + 1, j - hp : j + hp + 1]
            mat[i, j] = window_Response(Ix, Iy, k=k, method=method)
    return mat[hp:-hp, hp:-hp]


def non_MaximalSupression(img: np.ndarray, hp: int = 1) -> np.ndarray:
    img = image_padding(img=img, half_pad=hp)
    f = np.copy(img)
    n, m = img.shape
    for i in range(hp, n - hp):
        for j in range(hp, m - hp):
            submat = img[i - hp : i + hp + 1, j - hp : j + hp + 1]
            if img[i, j] == np.max(submat):
                f[i, j] = img[i, j] * 5
            else:
                f[i, j] = img[i, j] / 5
    return f
