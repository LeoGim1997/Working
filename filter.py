from function import gaussian_kernel, gaussian_kernel_basis
import numpy as np

def gaussian_blur(img : np.array, sigma :float = 1 ,fill_pad = False) -> np.array:
    '''
    Apply a 7x7 gaussian blur on a img (default)
    on 1 channel of the input image
    '''
    filter = gaussian_kernel(sigma)
    n,m = np.shape(img)
    cut = np.shape(filter)
    half_witdh = cut[0]//2
    #pad the image for fitting
    img_c = np.zeros((n+half_witdh,m+half_witdh))
    img_c[half_witdh:,half_witdh:] = img
    #filling the 4 corner with the same corner value as the original image
    if fill_pad:
        #top block of the matrix
        img_c[0:half_witdh,0:half_witdh] = img[0,0]*img_c[0:half_witdh,0:half_witdh]
        for i in range(half_witdh):
            img_c[i,half_witdh:-half_witdh] = img[0,:-half_witdh]
        img_c[0:half_witdh,-half_witdh:] = img[0,-1]*img_c[0:half_witdh,-half_witdh:]
        #bottom block of the matrix
        img_c[-half_witdh:,0:half_witdh] = img[-1,0]*img_c[-half_witdh:,0:half_witdh]
        for i in range(1,half_witdh):
            img_c[-i,half_witdh:-half_witdh] = img[-1,:-half_witdh]
        img_c[-half_witdh:,-half_witdh:] = img[-1,-1]*img_c[-half_witdh:,-half_witdh:]

        #left and right block of the matrix
        img_c[half_witdh:,0:half_witdh] = img[:,0:half_witdh]
        img_c[half_witdh:,-half_witdh:] = img[:,-half_witdh:]

    for i in range(half_witdh,n-half_witdh):
        for j in range(half_witdh,m-half_witdh-1):
                sub_matrix = img_c[i-half_witdh:i+half_witdh+1,j-half_witdh:j+half_witdh+1]
                prod = np.multiply(sub_matrix,filter)
                img_c[i,j] = np.average(prod)
    return img_c
