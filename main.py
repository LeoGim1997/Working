import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from im import normalize,integral_image_recursive_1channel


if __name__ == '__main__':
    img = mpimg.imread('/Users/leogimenez/Desktop/git_depo_local/Working/image/lena_gray.gif')
    img_norm = normalize(img)
    integral_img = integral_image_recursive_1channel(img_norm[:,:,0])
    plt.imshow(integral_img)
    plt.show()

