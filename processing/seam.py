import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from im import fast_rgb2grey
from function import compute_gradient

img = plt.imread('/Users/leogimenez/Desktop/git_depo_local/Working/image/image_folder/tower.png')
img_g = compute_gradient(fast_rgb2grey(img))

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
plt.figure()
plt.imshow(seam_mat)
plt.show()
