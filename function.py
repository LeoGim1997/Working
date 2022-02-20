import enum
from math import sqrt,pi,floor
from random import gauss
from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import convolve2d
from mpl_toolkits.mplot3d import Axes3D


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
    kernel = gaussian_mat[center_x-cut-1:center_x+cut,center_y-cut-1:center_y+cut]
    scale_factor = 1/np.average(kernel)
    kernel = scale_factor*kernel
    return kernel



def compute_gradient(img: np.array , operator = 'Sobel',return_xy_gradient: bool = False)-> np.array:
    '''
    Function to compute the gradient of the image according
    to certain kernel operator (default is Sobel operator)    
    '''
    if operator == 'Sobel':
        a1 = np.matrix([1, 2, 1])
        a2 = np.matrix([-1, 0, 1])
        Kx = a1.T * a2
        Ky = a2.T * a1
    if operator == 'Robert':
        Kx = np.array([[1,0],[0,-1]])
        Ky = np.array([[0,1],[-1,0]])
    # Apply the Sobel operator
    Gx = convolve2d(img, Kx, "same", "symm")
    Gy = convolve2d(img, Ky, "same", "symm")
    G = np.abs(Gx) + np.abs(Gy)
    if return_xy_gradient:
        return G,Gx,Gy
    return G 







