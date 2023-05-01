import numpy as np
from processing.filter import image_padding


def Q1(mat: np.ndarray, hp: int = 2) -> np.ndarray:
    return mat[0 : hp + 1, 0 : hp + 1]


def Q2(mat: np.ndarray, hp: int = 2) -> np.ndarray:
    return mat[0 : hp + 1, hp:]


def Q3(mat: np.ndarray, hp: int = 2) -> np.ndarray:
    return mat[hp:, hp:]


def Q4(mat: np.ndarray, hp: int = 2) -> np.ndarray:
    return mat[hp:, 0 : hp + 1]


def filterResponse(mat: np.ndarray, hp: int = 2) -> float:
    mapDict = {1: Q1, 2: Q2, 3: Q3, 4: Q4}
    mat = np.asarray(mat)
    n, m = mat.shape
    if n != m:
        raise ValueError(f"Input ndarray is not square.")
    if n == m != 2 * hp + 1:
        raise ValueError(f"Input ndarray have wrong dim")
    stdlist = [np.std(m) for m in (Q1(mat, hp), Q2(mat, hp), Q3(mat, hp), Q4(mat, hp))]
    max_sdt = np.argmin(stdlist)
    selectedQ = mapDict.get(max_sdt + 1)
    return np.mean(selectedQ(mat), dtype=np.float16)


def kuwaharaConvolution(img: np.ndarray, hp: int = 2) -> np.ndarray:
    p_img = image_padding(img, half_pad=hp, fill_pad=False, defaultfill=0)
    img = np.zeros(np.shape(p_img))
    N, M = np.shape(img)
    for i in range(hp, N - hp):
        for j in range(hp, M - hp):
            sub_matrix = p_img[i - hp : i + hp + 1, j - hp : j + hp + 1]
            img[i, j] = filterResponse(sub_matrix, hp)
    return img[hp:-hp, hp:-hp]


def kuwahara_response(img: np.ndarray, hp: int = 2) -> np.ndarray:
    shape = img.shape
    if len(shape) > 2:
        a = np.copy(img)
        for i in range(shape[-1]):
            a[..., i] = kuwaharaConvolution(img=img[..., i], hp=hp)
        return a
    return kuwaharaConvolution(img, hp=hp)
