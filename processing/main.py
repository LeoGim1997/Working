import numpy as np
from im import MyImage
from im import fast_rgb2grey
from gradient_filtering import compute_gradient
from filter import gaussian_blur
if __name__ == '__main__':
    a = MyImage('lena').get_matrix('/Users/leogimenez/Desktop/git_depo_local/Working/image/image_folder/Valve.PNG')
    a = fast_rgb2grey(a)
    MyImage.show_compare(compute_gradient(a, 'canny'), compute_gradient(a, 'sobel'))
