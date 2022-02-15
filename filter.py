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
    hw = cut[0]//2
    #pad the image for fitting
    img_c = np.zeros((n+2*hw,m+2*hw))
    img_c[hw:-hw,hw:-hw] = img
    #filling the 4 corner with the same corner value as the original image
    if fill_pad:
        #top block of the matrix
        img_c[0:hw,0:hw] = img[0,0]*img_c[0:hw,0:hw]
        for i in range(hw):
            img_c[i,hw:-hw] = img[0,:-hw]
        img_c[0:hw,-hw:] = img[0,-1]*img_c[0:hw,-hw:]
        #bottom block of the matrix
        img_c[-hw:,0:hw] = img[-1,0]*img_c[-hw:,0:hw]
        for i in range(1,hw):
            img_c[-i,hw:-hw] = img[-1,:-hw]
        img_c[-hw:,-hw:] = img[-1,-1]*img_c[-hw:,-hw:]

        #left and right block of the matrix
        img_c[hw:,0:hw] = img[:,0:hw]
        img_c[hw:,-hw:] = img[:,-hw:]
    N,M = np.shape(img_c)
    for i in range(hw,N-hw):
        for j in range(hw,M-hw-1):
                sub_matrix = img_c[i-hw:i+hw+1,j-hw:j+hw+1]
                prod = np.multiply(sub_matrix,filter)
                img_c[i,j] = np.average(prod)
    return img_c[hw:-hw,hw:-hw]
