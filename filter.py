from function import gaussian_kernel_basis
import numpy as np

def gaussian_blur(img : np.array ) -> np.array:
    '''
    Apply a 7x7 gaussian blur on a img (default)
    on 1 channel of the input image
    '''
    filter = gaussian_kernel_basis()
    n,m = np.shape(img)
    #pad the image for fitting
    img_c = np.ones((n+3,m+3))
    img_c[3:,3:] = img
    #filling the 4 corner with the same corner value as the original image

    #top block of the matrix
    img_c[0:3,0:3] = img[0,0]*img_c[0:3,0:3]
    for i in range(3):
        img_c[i,3:-3] = img[0,:-3]
    img_c[0:3,-3:] = img[0,-1]*img_c[0:3,-3:]
    #bottom block of the matrix
    img_c[-3:,0:3] = img[-1,0]*img_c[-3:,0:3]
    for i in range(1,3):
        img_c[-i,3:-3] = img[-1,:-3]
    img_c[-3:,-3:] = img[-1,-1]*img_c[-3:,-3:]

    #left and right block of the matrix
    img_c[1:-2,0:3] = img[:,0:3]
    img_c[1:-2,-3:] = img[:,-3:]

    #lower-left corner and bottom row
    for i in range(3,n-4):
        for j in range(3,m-4):
                sub_matrix = img[i-3:i+4,j-3:j+4]
                prod = np.multiply(sub_matrix,filter)
                img_c[i,j] = np.average(prod)
    return img_c[3:-4,4:-4]
