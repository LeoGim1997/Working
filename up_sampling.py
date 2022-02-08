import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from im import fast_rgb2grey

def naive_upsampling(img : np.array , new_dim1: int , new_dim2 :int) -> np.array:
    dim = np.shape(img)
    N,M = dim[0]*new_dim1 , dim[1]*new_dim2
    new_img = np.zeros((N,M))
    new_img[0::new_dim1,0::new_dim2] = img[:,:]

    for i in range(0,N):
        for j in range(1,M-1):
            current_pixel = new_img[i,j]
            if current_pixel != 0:
                new_img[i:new_dim1,j:new_dim2] = new_img[i,j]
    return new_img


