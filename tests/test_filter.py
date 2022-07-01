import pytest
import sys
from pathlib import Path
import numpy as np
sys.path.append(Path(__file__).parents[1].as_posix())
from processing.filter import image_padding


def imageBlank(shape: int = 10, coeff=0):
    if coeff == 1:
        return np.ones((shape, shape))
    return np.zeros((shape, shape))


def genericImage():
    m = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    return np.asarray(m)


@pytest.mark.parametrize('input', [imageBlank()])
def test_imagePadding1(input):
    assert isinstance(image_padding(input), (np.generic, np.ndarray))


@pytest.mark.parametrize('img', [imageBlank(4)])
def test_imagePadding2(img):
    a = image_padding(img)
    assert a.shape[0] == a.shape[1]


@pytest.mark.parametrize('img', [imageBlank(4)])
def test_imagePadding3(img):
    a = image_padding(img, half_pad=2)
    assert a.shape[0] == img.shape[0] + 4


@pytest.mark.parametrize('img', [imageBlank(2)])
def test_imagePadding4(img):
    a = image_padding(img, half_pad=1)
    assert a.shape == (img.shape[0] + 2, img.shape[1] + 2)


@pytest.mark.parametrize('img', [genericImage()])
def test_imagePadding5(img):
    a = image_padding(img, half_pad=1, fill_pad=True)
    assert a[0, 0] == 1
    assert a[2, 0] == a[2, 0]


@pytest.mark.parametrize('img', [genericImage()])
def test_imagePadding6(img):
    a = image_padding(img, half_pad=1, fill_pad=True)
    assert a[0, 0] == 1
    assert a[2, 0] == a[2, 0]


PADDED1 = np.array([[1., 1., 1., 1.],
                   [1., 0., 0., 1.],
                   [1., 0., 0., 1.],
                   [1., 1., 1., 1.]])


@pytest.mark.parametrize('img , expected', [(imageBlank(2), PADDED1)])
def test_imagePadding7(img, expected):
    f = image_padding(img, half_pad=1, fill_pad=False)
    comparaison = f == expected
    assert comparaison.all()


PADDED2 = np.array([[1, 1, 2, 3, 3],
                    [1, 1, 2, 3, 3],
                    [4, 4, 5, 6, 6],
                    [7, 7, 8, 9, 9],
                    [7, 7, 8, 9, 9]])


@pytest.mark.parametrize('img , expected', [(genericImage(), PADDED2)])
def test_imagePadding8(img, expected):
    f = image_padding(img, half_pad=1, fill_pad=True)
    comparaison = f == expected
    assert comparaison.all()
