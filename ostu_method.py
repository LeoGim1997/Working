from re import A
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from histogram import img_histogram
from im import gray_scale


def otsu_method(img : np.array) -> int:
    '''
    Cherche un seuil dans une image NxM (1 channel)
    Returns:
        level : rang separant l'histogramme en 2
    '''
    dim = np.shape(img)
    if len(dim)>2:
        print('The image is multichannel performing on 1 channel')
        img = img[:,:,0]
    number,value = img_histogram(img,256)
    total = dim[0]*dim[1]
    top = 256
    sumB = 0.0
    wB = 0.0
    maximum = 0.0
    vec = np.linspace(0,top-1,top)
    sum1 = np.dot(vec,number)
    l_var = [0]
    for i in range(0,top-1):
        wF = total - wB
        if wB and wF>0:
            mF = (sum1-sumB)/float(wF)
            val = wB * wF * ((sumB / float(wB)) - mF) * ((sumB / float(wB)) - mF)
            l_var.append(val)
            if val >= maximum:
                level = i
                maximum = val
        wB = wB + number[i]
        sumB = sumB + (i-1)*number[i]
    return level,l_var


def perform_ostu_threshold(img : np.array):
    img_c = np.copy(img)
    level, l_var = otsu_method(img)
    for i,row in enumerate(img_c):
        for j,p in enumerate(row):
            p = 0 if p < level else 1
            img_c[i,j] = p
    return img_c


def generate_final_img(img: np.array):
    '''
    Genere une figure comparant l'image originale 
    et l'image seuil
    '''
    img_threshold = perform_ostu_threshold(img)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.title('Image originale')
    plt.subplot(1,2,2)
    plt.imshow(img_threshold,cmap='gray')
    plt.title('Image seuil')
    plt.show()





