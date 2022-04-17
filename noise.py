import numpy as np


def noise(image: np.array, noise_typ='s_p'):

    if len(np.shape(image)) == 3:
        row, col, ch = np.shape(image)
    else:
        row, col = np.shape(image)
        ch = 1
    if noise_typ == "gauss":
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s_p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
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


def salt_and_peper_v1(img_input, bmin=300, bmax=100000):
    """
    More the range is wide more the noise is wide.
    """

    img = np.copy(img_input)
    row, col = np.shape(img)
    number_of_pixels = np.random.randint(bmin, bmax)
    for i in range(number_of_pixels):
        y_coord = np.random.randint(0, row-1)
        x_coord = np.random.randint(0, col-1)
        img[y_coord, x_coord] = 255

    number_of_pixels = np.random.randint(bmin, bmax)
    for i in range(number_of_pixels):
        y_coord = np.random.randint(0, row-1)
        x_coord = np.random.randint(0, col-1)
        img[y_coord, x_coord] = 0
    return img
