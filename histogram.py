import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def img_histogram( img : np.array , nb_bins =100) -> np.array:
    '''
    Process the histogram of an 1 channel grayscale image
    Return:
        number (np.array) : occurences de chaque pixel
        value (np.array) : valeur associé a chaque occurence
    '''
    dim = np.shape(img)
    ntotal = dim[0]*dim[1]
    if len(dim)>2: 
        raise ValueError(f'The dimension of img should be 2 and is {dim}')
    value = np.reshape(img,ntotal)
    number,value = np.histogram(value,bins=nb_bins)
    return number,value

def plot_histogram_mutichannel(img: np.array, nbins = 100):
    dim = np.shape(img)
    ntotal = dim[0]*dim[1]
    plt.figure()
    c = ['red','green','blue']
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.hist(np.reshape(img[:,:,i],ntotal),bins=nbins,color=c[i])
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.title(f'channel {c[i]}')
    plt.show()
