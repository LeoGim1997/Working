import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from histogram import img_histogram


def otsu_method(img: np.array, return_hist=False) -> int:
    """
    Cherche un seuil dans une image NxM (1 channel)
    Returns:
        level : rang separant l'histogramme en 2
    """
    dim = np.shape(img)
    if len(dim) > 2:
        print("The image is multichannel performing on 1 channel")
        img = img[:, :, 0]
    number, value = img_histogram(img, 256)
    total = dim[0] * dim[1]
    top = 256
    sumB = 0.0
    wB = 0.0
    maximum = 0.0
    vec = np.linspace(0, top - 1, top)
    sum1 = np.dot(vec, number)
    l_var = []
    for i in range(0, top):
        wF = total - wB
        if wB and wF > 0:
            mF = (sum1 - sumB) / float(wF)
            val = wB * wF * ((sumB / float(wB)) - mF) * ((sumB / float(wB)) - mF)
            l_var.append(val)
            if val >= maximum:
                level = i
                maximum = val
        wB = wB + number[i]
        sumB = sumB + (i - 1) * number[i]
    if return_hist:
        return number, value, level, l_var
    return level, l_var


def perform_ostu_threshold(img: np.array):
    img_c = np.copy(img)
    level, l_var = otsu_method(img, return_hist=False)
    for i, row in enumerate(img_c):
        for j, p in enumerate(row):
            p = 0 if p < level else 1
            img_c[i, j] = p
    return img_c


def generate_graph(img: np.array):
    number, value, level, l_var = otsu_method(img, return_hist=True)
    number = np.concatenate((np.zeros(1), number), axis=0)
    dim = np.shape(img)
    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Occurences", color=color)
    l1 = ax1.plot(value, number, color=color, label="Histogramme")
    ax1.fill_between(value, number, alpha=1)
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:red"
    ax2.set_ylabel("variance", color=color)  # we already handled the x-label with ax1
    l2 = ax2.plot(l_var, color=color, label="Variance")
    ax2.tick_params(axis="y", labelcolor=color)
    plt.show()


def generate_final_img(img: np.array):
    """
    Genere une figure comparant l'image originale
    et l'image seuil
    """
    img_threshold = perform_ostu_threshold(img)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image originale")
    plt.subplot(1, 2, 2)
    plt.imshow(img_threshold, cmap="gray")
    plt.title("Image seuil")
    plt.show()
