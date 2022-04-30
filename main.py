import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from filter import median_filtering
from noise import noise


if __name__ == '__main__':
    img = mpimg.imread('/Users/leogimenez/Desktop/git_depo_local/Working/image/image_folder/Lena.jpeg')
    img_noise = noise(img, noise_type='s_p')
    restored_img = median_filtering(img_noise, size=3)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(img_noise, cmap='gray')
    plt.title('Noisy Image with Salt and Pepper noise')
    plt.subplot(1, 3, 3)
    plt.imshow(restored_img, cmap='gray')
    plt.title('Denoised Image using Median filtering')
    plt.savefig('output/output1.png')
