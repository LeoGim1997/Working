import pytest
from pathlib import Path
import sys
from base_test import Test
from processing.im import MyImage

from processing.kuwahara import kuwahara_response

@pytest.mark.parametrize('image', ['lena'])
def test_kuwahara_response(image):
    imageMatrix = MyImage(image).get_matrix()
    testName = f'{image}_kuwahara'
    processed = kuwahara_response(imageMatrix,hp=2)
    Test.saveOutput(image,processed,figName=testName)


