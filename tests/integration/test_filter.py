from filter import sobel_filter_horizontal
from im import MyImage
import pytest
from base_test import Test


@pytest.mark.parametrize('name',['lena'])
def test_filter_sobel_horizontal(name):
    image = MyImage(name).get_matrix()
    process = sobel_filter_horizontal(img=image)
    Test.saveOutput(image,process,figName=f"testSobel_{name}")
