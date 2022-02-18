import numpy as np
from scipy.signal import convolve2d
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
    img_c = image_padding(img,hw,fill_pad)
    N,M = np.shape(img_c)
    for i in range(hw,N-hw):
        for j in range(hw,M-hw-1):
                sub_matrix = img_c[i-hw:i+hw+1,j-hw:j+hw+1]
                prod = np.multiply(sub_matrix,filter)
                img_c[i,j] = np.average(prod)
    return img_c[hw:-hw,hw:-hw]


def image_padding(img : np.array , half_pad :int = 4,fill_pad =False) -> np.array:
    '''
    Function returning a padded image use for convolution
    with a square fitler of half_with = half_pad.
    The center of the image will be the orginal image
    '''
    n,m = np.shape(img)
    hw = half_pad
    #pad the image for fitting
    img_c = np.ones((n+2*hw,m+2*hw))
    img_c[hw:-hw,hw:-hw] = img

    #filling the 4 corner with the same corner value as the original image
    if fill_pad:
        #top block of the matrix
        img_c[0:hw,0:hw] = img[0,0]*img_c[0:hw,0:hw]
        for i in range(hw):
            img_c[i,hw:-hw] = img[0,:]
        img_c[0:hw,-hw:] = img[0,-1]*img_c[0:hw,-hw:]
        #bottom block of the matrix
        img_c[-hw:,0:hw] = img[-1,0]*img_c[-hw:,0:hw]
        for i in range(1,hw):
            img_c[-i,hw:-hw] = img[-i,:]
        img_c[-hw:,-hw:] = img[-1,-1]*img_c[-hw:,-hw:]

        #left and right block of the matrix
        img_c[hw:-hw,:hw] = img[:,:hw]
        img_c[hw:-hw,-hw:] = img[:,-hw:]
    return img_c

def sobel_filter_horizontal(img: np.array) -> np.array:
    '''
    Return the horizontal image gradient using Sobel convolution
    '''
    
    filter = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    Gx = image_padding(img,1,fill_pad=False)
    img = np.zeros(np.shape(Gx))
    N,M = np.shape(Gx)
    for i in range(1,N-1):
        for j in range(1,M-1):
            sub_matrix = Gx[i-1:i+2,j-1:j+2]
            prod = np.multiply(sub_matrix,filter)
            img[i,j] = np.sum(prod)
    return img[1:-1,1:-1]

def sobel_filter_vertical(img: np.array) -> np.array:
    '''
    Return the vertical image gradient using Sobel convolution
    '''

    filter = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    Gy = image_padding(img,1,fill_pad=False)
    img = np.zeros(np.shape(Gy))
    N,M = np.shape(Gy)
    for i in range(1,N-1):
        for j in range(1,M-1):
            sub_matrix = Gy[i-1:i+2,j-1:j+2]
            img[i,j] = np.sum(np.multiply(sub_matrix,filter))
    return img[1:-1,1:-1]

def normalized_sobel_filter(img : np.array , threshold : int = 0) -> np.array:
    Gx = sobel_filter_horizontal(img)
    Gy = sobel_filter_vertical(img)
    n,m = np.shape(img)
    img_f = np.zeros((n,m))
    img_f = np.abs(Gx)+np.abs(Gy)
    if threshold != 0:
        for i in range(n):
            for j in range(m):
                if img_f[i,j]<threshold:
                    img_f[i,j] = 0
                else:
                    img_f[i,j] = 300
    return img_f


def Sobel_2(img: np.array) -> np.array:
    a1 = np.matrix([1, 2, 1])
    a2 = np.matrix([-1, 0, 1])
    Kx = a1.T * a2
    Ky = a2.T * a1

    # Apply the Sobel operator
    Gx = convolve2d(img, Kx, "same", "symm")
    Gy = convolve2d(img, Ky, "same", "symm")
    G = np.abs(Gx) + np.abs(Gy)
    return G,Gx,Gy