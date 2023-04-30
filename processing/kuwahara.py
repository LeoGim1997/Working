import numpy as np
from processing.filter import image_padding
from processing.im import MyImage
from typing import Iterator, Tuple


def Q1(mat: np.ndarray) -> np.ndarray:
    return mat[0:3, 0:3]


def Q2(mat: np.ndarray) -> np.ndarray:
    return mat[0:3, 2:]


def Q3(mat: np.ndarray) -> np.ndarray:
    return mat[2:, 2:]


def Q4(mat: np.ndarray) -> np.ndarray:
    return mat[2:, 0:3]


def filterResponse(mat: np.ndarray) -> float:
    mapDict = {1: Q1, 2: Q2, 3: Q3, 4: Q4}
    mat = np.asarray(mat)
    n, m = mat.shape
    if n != m:
        raise ValueError(f"Input ndarray is not square.")
    if n == m != 5:
        raise ValueError(f"Input ndarray have wrong dim")
    stdlist = [np.std(m) for m in (Q1(mat), Q2(mat), Q3(mat), Q4(mat))]
    max_sdt = np.argmin(stdlist)
    selectedQ = mapDict.get(max_sdt + 1)
    return np.mean(selectedQ(mat),dtype=np.float16)


def kuwaharaConvolution(img: np.ndarray, hp: int = 2) -> np.ndarray:
    p_img = image_padding(img, half_pad=hp, fill_pad=False, defaultfill=0)
    img = np.zeros(np.shape(p_img))
    N, M = np.shape(img)
    for i in range(hp, N - hp):
        for j in range(hp, M - hp):
            sub_matrix = p_img[i - hp : i + hp + 1, j - hp : j + hp + 1]
            img[i, j] = filterResponse(sub_matrix)
    return img[hp:-hp, hp:-hp]


def kuwahara_response(img: np.ndarray, hp: int = 2) -> np.ndarray:
    shape = img.shape
    if len(shape) > 2:
        a = np.copy(img)
        for i in range(shape[-1]):
            a[..., i] = kuwaharaConvolution(img=img[..., i], hp=hp)
        return a
    return kuwaharaConvolution(img, hp=hp)
