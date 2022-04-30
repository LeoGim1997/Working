import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from filter import gaussian_blur
from function import plot_2_image
from histogram import plot_hist, plot_histogram_mutichannel
from im import normalize,fast_rgb2grey


if __name__ == '__main__':
    img = mpimg.imread('/Users/leogimenez/Desktop/git_depo_local/Working/image/image_folder/lena_couleur.jpeg')
    img = fast_rgb2grey(img)
    img_g = gaussian_blur(img)
    plot_2_image(img,img_g)
    
