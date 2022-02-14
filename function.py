from math import sqrt,pi,floor
from random import gauss
from turtle import width
import numpy as np
import matplotlib.pyplot as plt


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

def gaussian_sample(sigma:float = 1 ,min:int = -3,max: int = 3,n_samples : int= 50) -> np.array:
    '''
    Return a 1-d Gaussian array of std sigma
    '''
    coeff = 1/np.sqrt(sigma**2)
    coeff_pi = coeff/np.sqrt(2*pi)
    x = np.linspace(min,max,n_samples)
    x = coeff_pi*np.exp(-coeff*pow(x,2))
    return np.reshape(x,(len(x),1)) 


def gaussian_kernel(sigma :float = 1)-> np.array:
    '''
    Return normalized gaussian kernel of std sigma
    of half-width h=3*sigma
    '''
    if sigma<0:
        raise ValueError('The std cannot be negative')
    half_w = floor(sigma)*3
    witdh = 2*half_w+1
    # add more point to the mesh-grid
    x = gaussian_sample(sigma,-half_w,half_w,witdh*5)
    shape_x = np.shape(x)[0]
    y = np.copy(x)
    gaussian_mat = np.dot(x,np.transpose(x))
    scale_factor = 1/np.average(gaussian_mat)
    gaussian_mat = scale_factor*gaussian_mat
    #cropping the image to only get usefull value
    n,m = np.shape(gaussian_mat)
    center_x = witdh*5//2
    center_y = witdh*5//2
    cut = witdh*5//5
    kernel = gaussian_mat[center_x-cut-1:center_x+cut,center_y-cut:center_y+cut]
    return kernel

def plot_matrix(mat : np.array):
    '''
    Default settings for plotting a matrix of reasonable size
    in case of large matrix better use imshow() from mpimp module
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


