import numpy as np
from filter import image_padding
from im import MyImage


def Q1(mat): return mat[0:3, 0:3]
def Q2(mat): return mat[0:3, 2:]
def Q3(mat): return mat[2:, 2:]
def Q4(mat): return mat[2:, 0:3]


MAP_DICT = {
    1: Q1,
    2: Q2,
    3: Q3,
    4: Q4
}


def generateQMatrix(mat: np.array):
    """Allows to generate
    """
    return (MAP_DICT.get(k)(mat) for k in range(1, 5))


def filterResponse(mat: np.array) -> float:
    mat = np.asarray(mat)
    n, m = mat.shape
    if n != m:
        raise ValueError(f'Input ndarray is not square.')
    if n == m != 5:
        raise ValueError(f'Input ndarray have wrong dim')
    stdlist = [np.std(m) for m in generateQMatrix(mat)]
    max_sdt = np.argmin(stdlist)
    selectedQ = MAP_DICT.get(max_sdt + 1)
    return np.mean(selectedQ(mat))


def kuwaharaConvolution(img: np.array, hp: int = 2) -> np.array:
    p_img = image_padding(img, half_pad=hp, fill_pad=False, defaultfill=0)
    img = np.zeros(np.shape(p_img))
    N, M = np.shape(img)
    for i in range(hp, N - hp):
        for j in range(hp, M - hp):
            sub_matrix = p_img[i - hp:i + hp + 1, j - hp:j + hp + 1]
            img[i, j] = filterResponse(sub_matrix)
    return img[hp:-hp, hp:-hp]


def kuwahara_response(img: np.array, hp: int = 2) -> np.array:
    shape = img.shape
    if len(shape) >= 2:
        a = np.copy(img)
        for i in range(shape[-1]):
            a[..., i] = kuwaharaConvolution(img=img[..., i], hp=hp)
        return a
    return kuwaharaConvolution(img, hp=hp)
