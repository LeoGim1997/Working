import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def noise(image: np.array, noise_typ='s_p'):

    if len(np.shape(image)) == 3:
        row, col, ch = np.shape(image)
    else:
        row, col = np.shape(image)
        ch = 1
    if noise_typ == "gauss":
        gauss = gaussian_noise(image, 5, 14)
        return image + gauss  # Additive noise
    elif noise_typ == "s_p":
        return salt_and_peper_noise(image, alpha=0.3)
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def salt_noise(img: np.array, alpha=0.8):
    """
    Generate a noisy image with a salt-paper noise with
    a percentage alpha of pixels affected.
    The affection of noise value is done as follow:
        - Creatio of matrix M of same dimension as input matrix.\\
        - M is filled with probability values from a uniform law.\\
        - We note p(x) the probability of pixel x, and I(x) it intensity:\\
            if p(x) in [0,alpha[ then I(x) = 0.\\
            if p(x) in [alpha/2,alpha] then I(x) = Max(I) (Max intentisity of input image). \\
            if p(x) in ]alpha,1] then I(x) is unchanged.
    Args:
        img (np.array): input 1 channel grayscale image.
        alpha (np.array): percentage of pixel affected by noise.
    Returns:
        img_final (np.array): Image noised.
    """
    n, m = np.shape(img)
    max = np.max(img)
    img_final = np.copy(img)
    img_noise = np.random.uniform(0, 1, (n, m))
    for i in range(n):
        for j in range(m):
            if (img_noise[i, j] < alpha):
                img_final[i, j] = 0
            if (img_noise[i, j] < alpha / 2) and (img_noise[i, j] <= alpha):
                img_final[i, j] = max
    return img_final


def gaussian_noise(img: np.array, mean: float = 0, sigma: float = 1) -> np.array:
    """
    Generate an additive gaussian noise image with `mu`=mean and `std`=sigma.
    Args:
        img (np.array): input image of size n,m.
        mean (float): mean of the gaussian distribution.
        sigma (float): standard deviation of the gaussian distribution.
    Returns:
        gauss (np.array): gaussian noise samples matrix.
    """
    n, m = np.shape(img)
    gauss = np.random.normal(mean, sigma, (n, m))
    return gauss
