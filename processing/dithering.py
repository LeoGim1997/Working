import numpy as np
from im import MyImage
import itertools as it
from filter import image_padding

# On load une image
Img = MyImage("david")
img = Img.get_matrix()
img = img * 255
print(img)


def findcloset(pixel: float, color: float = 255):
    return round(pixel / color)


def dithering(img: np.array) -> np.array:
    """
    Use the Floyd-Steinberg algorithm to dither an image.
    """
    hp = 1
    imgc = np.copy(img)
    imgprocess = image_padding(imgc, half_pad=hp, fill_pad=False)
    n, m = np.shape(img)
    arr = np.array(imgprocess, dtype=float) / 255
    for i, j in it.product(range(n), range(m)):
        old_pixel = imgprocess[i, j].copy()
        new_pixel = findcloset(old_pixel)
        imgprocess[i, j] = new_pixel
        # we need to find the 2 local max inside
        # the pixel neighborhood
        quant_error = old_pixel - new_pixel
        imgprocess[i + 1, j] = imgprocess[i + 1, j] + quant_error * 7 / 16
        imgprocess[i - 1, j + 1] = imgprocess[i - 1, j + 1] + quant_error * 3 / 16
        imgprocess[i, j + 1] = imgprocess[i, j + 1] + quant_error * 5 / 16
        imgprocess[i + 1, j + 1] = imgprocess[i + 1, j + 1] + quant_error * 1 / 16
    return imgprocess[1:-1, 1:-1]


# pixels[x + 1][y    ] := pixels[x + 1][y    ] + quant_error × 7 / 16
# pixels[x - 1][y + 1] := pixels[x - 1][y + 1] + quant_error × 3 / 16
# pixels[x    ][y + 1] := pixels[x    ][y + 1] + quant_error × 5 / 16
# pixels[x + 1][y + 1] := pixels[x + 1][y + 1] + quant_error × 1 / 16
MyImage.show_compare(img, dithering(img))
