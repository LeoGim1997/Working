from math import exp,sqrt,pi
import numpy as np
import matplotlib.pyplot as plt



def gaussian_2d(min = -3,max = 3,sigma = 0.5) -> np.array:
    
    coeff = 1/sqrt(float(2*pi*sigma**2))
    X = np.arange(min, max, 0.25)
    Y = np.arange(min, max, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = coeff*np.exp(-X**2 - Y**2)/(2*sigma**2)
    return X,Y,Z

def gaussian_kernel_basis() -> np.array:
    '''
    Return a 7x7 gaussian kernel np.array
    of sigma 1
    '''
    kernel = np.array([
    [0,0,0,0,0,0,0],
    [0,0,0.01,0.01,0.01,0,0],
    [0,0.01,0.05,0.11,0.05,0.01,0],
    [0,0.01,0.11,0.25,0.11,0.01,0],
    [0,0.01,0.05,0.11,0.05,0.01,0],
    [0,0,0.01,0.01,0.01,0,0],
    [0,0,0,0,0,0,0]])
    scale = 1/np.average(kernel)
    return scale*kernel

def plot_matrix(mat : np.array):
    '''
    Default settings for plotting a matrix
    '''
    fig, ax = plt.subplots()
    ax.matshow(mat, cmap='viridis')
    for (i, j), z in np.ndenumerate(mat):
        ax.text(j, i, '{:0.01f}'.format(z), ha='center', va='center')
    plt.show()


def plot_2_image( img : np.array , img_processed : np.array , icmap = 'gray'):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img,cmap=icmap)
    plt.subplot(1,2,2)
    plt.imshow(img_processed , cmap= icmap)
    plt.show()


