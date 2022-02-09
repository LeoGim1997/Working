import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from histogram import plot_hist, plot_histogram_mutichannel
from im import *


if __name__ == '__main__':
    img = mpimg.imread('/Users/leogimenez/Desktop/git_depo_local/Working/image/image_folder/lena_couleur.jpeg')
    img = fast_rgb2grey(img)
    plot_hist(img,show_im=True)
