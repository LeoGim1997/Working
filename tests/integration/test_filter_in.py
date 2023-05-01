from filter import (sobel_filter_horizontal,
                    sobel_filter_vertical,
                    normalized_sobel_filter)
from im import MyImage
import pytest
from base_test import Test


@pytest.mark.parametrize('name',['lena'])
def test_filter_sobel_horizontal(name):
    image = MyImage(name).get_matrix()
    process = sobel_filter_horizontal(img=image)
    Test.saveOutput(image,process,figName=f"testSobel_{name}_horizontal")

@pytest.mark.parametrize('name',['lena'])
def test_filter_sobel_vertical(name):
    image = MyImage(name).get_matrix()
    process = sobel_filter_vertical(img=image)
    Test.saveOutput(image,process,figName=f"testSobel_{name}_vertical")

@pytest.mark.parametrize('name',['lena'])
def test_filter_sobel_norm(name):
    image = MyImage(name).get_matrix()
    process = normalized_sobel_filter(img=image)
    Test.saveOutput(image,process,figName=f"testSobel_{name}_norm")
