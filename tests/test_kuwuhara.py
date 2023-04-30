import pytest
from pathlib import Path
import sys
from base_test import Test
from processing.im import MyImage

from processing.kuwahara import kuwahara_response

@pytest.mark.parametrize('image', [MyImage('lena').get_matrix()])
def test_kuwahara_response(image):
    testName = 'Image Lena'
    processed = kuwahara_response(image,hp=2)
    Test.saveOutput(image,processed,figname=testName)

