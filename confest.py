# Generate Image for testing
import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image
import pytest
a = '/Users/leogimenez/Desktop/git_depo_local/Working/image/image_folder/Lena.jpeg'


@pytest.fixture(scope="session")
def lena_standard():
    img = Path(__file__).parent / 'image_folder/Lena.jpeg'

    return mpimg.imread(img.as_posix())
