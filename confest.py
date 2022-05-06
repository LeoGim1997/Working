# Generate Image for testing
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import matplotlib.pyplot as plt


@pytest.fixture(scope="session")
def lena_standard():
    img = Path(__file__).parent / 'image_folder/Lena.jpeg'

    return mpimg.imread(img.as_posix())


@pytest.fixture(scope="session")
def simple_path():
    np.random.seed(5)
    size_row = 101
    mat = np.zeros((size_row, size_row))
    origin = 20
    chemin = np.random.randint(0, 3, size=size_row)
    for c, row in enumerate(mat):
        if c == 0:
            row[origin] = 1
            continue
        if c == size_row - 1:
            continue
        else:
            incr = chemin[c]
            row[origin - (incr + 1)] = 1
            origin = origin - (incr + 1)
    return mat
