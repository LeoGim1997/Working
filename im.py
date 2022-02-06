
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
img = mpimg.imread('/Users/leogimenez/Desktop/git_depo_local/Working/image/lena_gray.gif')


#Basic image processing
def remove_channels(img : np.array , channels : int) -> np.array:
    img_c = np.copy(img)
    shape = np.shape(img)
    width = shape[0]
    height = shape[1]
    for i in range(width):
        for j in range(height):
            img_c[i][j][channels] = 0
    return img_c

def plot_channel():
    plt.figure()
    for i in range(0,3):
        im = remove_channels(img,i)
        plt.subplot(1,3,i+1)
        plt.title(f'channel num {i}=0')
        plt.imshow(im)
    plt.show()


def addition_component(img : np.array):
    img_c = np.copy(img)
    shape = np.shape(img)
    width = shape[0]
    height = shape[1]
    for i in range(1,width-1):
        for j in range(1,height-1):
            img_c[i][j] = img_c[i][j]+img_c[i-1][j-1]
    return img_c


def normalize(img :np.array) -> np.array:
   return (img - np.min(img)) / (np.max(img) - np.min(img))


def integral_image_recursive_1channel(img : np.array) ->np.array:
    shape = np.shape(img)
    width = shape[0]
    height = shape[1]
    I = np.zeros((width,height))
    #Init
    I[0,0] = img[0,0]
    for i in range(1,width):
        for j in range(1,height):
            I[i,j] = img[i,j]+I[i,j-1]+I[i-1,j]-I[i-1,j-1]
    return I








