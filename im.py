
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

def gray_scale_conv_1channel(img : np.array , channel = 0) -> np.array:
    img_1c = normalize(img[:,:,channel])
    for row in img_1c:
        for pixel in row:
            if pixel<= 0.04045:
                pixel = pixel/float(12.92)
            if pixel > 0.04045:
                pixel = ((pixel+0.055)/float(1.055))**(2.4)
    return img_1c

def gray_scale(img : np.array) -> np.array:
    norm = [0.2126,0.7152,0.0722]
    dim = np.shape(img)
    img2 = np.zeros((dim[0],dim[1]))
    l_matrix = [value*gray_scale_conv_1channel(img,channel) for channel,value in enumerate(norm) ]
    for i in range(dim[0]):
        for j in range(dim[1]):
            p = l_matrix[0][i,j]+l_matrix[1][i,j]+l_matrix[2][i,j]
            img2[i,j] = 12.92*p if p <= 0.0031308 else (((p+0.055)/1.055))**(2.4)
    img_finale = np.ones((dim[0],dim[1],dim[2]+1))
    for i in range(3):
        img_finale[:,:,i] = img2
    return img_finale


   







    







