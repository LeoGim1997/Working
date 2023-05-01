import numpy as np


def up_sampling_nn(img: np.ndarray, factor: int = 3) -> np.ndarray:
    """
    This function allows to up-sample the inpute image
    using nearest-neighboors averaging technique.

    Args:
        img (arraylike) : input image.
        factor (int) : Resize factor.
    Returns:
        ndarray : factor resized image.
    """
    if factor % 2 == 0:
        raise ValueError(
            f"resize factor must be even for correct indexing. Input was {factor}"
        )
    n, m = np.shape(img)
    img_2 = np.zeros((n * factor, m * factor))
    mat = np.ones((factor, factor))
    step = int((factor - 1) / 2)
    for i in range(n):
        for j in range(m):
            centerx, centery = factor * i, factor * j
            if centerx == (n * 3) - 1 or centery == (m * 3) - 1:
                continue
            img_2[centerx : centerx + factor, centery : centery + factor] = (
                mat * img[i, j]
            )
    return img_2
