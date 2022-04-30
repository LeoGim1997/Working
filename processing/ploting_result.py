import numpy as np
import matplotlib.pyplot as plt
from function import compute_gradient 
from im import remove_channels

def plot_Sobel_processing(img : np.array, icmap: str = 'gray')-> np.array:
    G,Gx,Gy = compute_gradient(img,operator='Sobel',return_xy_gradient=True)
    title = ['orginal image','Sobel filtered image','Gx gradient','Gy gradient']
    plt.figure()
    for c,(img,t) in enumerate(zip([img,G,Gx,Gy],title)):
        plt.subplot(1,len(title),c+1)
        plt.imshow(img,cmap = icmap)
        plt.title(t)
    plt.show()

def plot_channel(img : np.array):
    plt.figure()
    for i in range(0,3):
        im = remove_channels(img,i)
        plt.subplot(1,3,i+1)
        plt.title(f'channel num {i}=0')
        plt.imshow(im)
    plt.show()

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


def image_3d_plot(img: np.array):
    '''
    3d representation of an input image
    using the gray-scale level intensity as the z-axis
    '''
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, img ,rstride=1, cstride=1, cmap=plt.cm.gray,linewidth=0)
    plt.show()