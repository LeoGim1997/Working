import numpy as np
from gradient_filtering import compute_gradient
from im import MyImage
from genericImage import opposite_eye
import matplotlib.pyplot as plt
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


def haarisResponse(k: float = 0.004,
                   num: int = 100,
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


def showResponse() -> None:
    plt.figure()
    plt.imshow(haarisResponse(num=500, revert=False),
               cmap='gray',
               origin='lower',
               extent=(0, 1, 0, 1))
    plt.xlabel('\u03BB1')
    plt.ylabel('\u03BB2')
    plt.title('Response for Haaris corner detector.')
    plt.show()
