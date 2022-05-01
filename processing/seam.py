import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from im import fast_rgb2grey
from function import compute_gradient

img = plt.imread('/Users/leogimenez/Desktop/git_depo_local/Working/image/image_folder/tower.png')
img_g = compute_gradient(fast_rgb2grey(img))


def compute_energy(img_g: np.array) -> np.array:
    n, m = np.shape(img_g)
    seam_mat = np.copy(img_g)
    seam_mat[0, :] = img_g[0, :]
    for i in range(1, n):
        for j in range(m):
            if j == 0:
                if seam_mat[i - 1, j] < seam_mat[i - 1, j + 1]:
                    seam_mat[i, j] += seam_mat[i - 1, j]
                else:
                    seam_mat[i, 0] += seam_mat[i - 1, j + 1]
            elif j == m - 1:
                if seam_mat[i - 1, -1] < seam_mat[i - 1, -2]:
                    seam_mat[i, j] += seam_mat[i - 1, -1]
                else:
                    seam_mat[i, j] += seam_mat[i - 1, -2]
            else:
                above_pixel = seam_mat[i - 1, j - 1:j + 2]
                rank = np.argmin(above_pixel)
                seam_mat[i, j] += seam_mat[i - 1, j - (rank - 1)]
    return seam_mat


seam = compute_energy(img_g)


def compute_path(seam, index):
    neighbor = seam[:, index - 1:index + 2]
    n, m = np.shape(seam)
    path_coor = []
    for i in np.arange(-2, -n + 1, -1):
        pixel_coor = np.argmin(neighbor[i])
        path_coor.append((i, index - (pixel_coor - 1)))
    return path_coor


def compute_all_path(seam: np.array, tshld: int) -> dict:
    last_line = list(seam[-1, :])
    l_index = list(map(lambda x: last_line.index(x) if x < tshld else None, last_line))

    dict_final = dict()
    for c, index in enumerate(l_index):
        if (index is None) or (index < 4) or (index > (seam.shape[1] - 4)):
            continue
        dict_final.update({index: compute_path(seam, index)})
    return dict_final
