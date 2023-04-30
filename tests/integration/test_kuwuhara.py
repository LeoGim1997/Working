import pytest
from pathlib import Path
import sys
from base_test import Test
from processing.im import MyImage

from processing.kuwahara import kuwahara_response

@pytest.mark.integration
@pytest.mark.parametrize('image', ['lena'])
def test_kuwahara_response(image):
    imageMatrix = MyImage(image).get_matrix()
    testName = f'{image}_kuwahara'
    processed = kuwahara_response(imageMatrix,hp=2)
    Test.saveOutput(imageMatrix,processed,figName=testName)


@pytest.mark.integration
@pytest.mark.parametrize('image',['lena_couleur'])
def test_kuwahara_couleur(image):
    imageMatrix = MyImage(image).get_matrix()
    testName = f'{image}_kuwahara'
    processed = kuwahara_response(imageMatrix,hp=2)
    Test.saveOutput(imageMatrix,processed,figName=testName)

@pytest.mark.integration
@pytest.mark.parametrize('halfpad',(2,4,5,9,15))
def test_multiple_halpad(halfpad):
    imageMatrix = MyImage('david').get_matrix()
    testName = f'lena_kuwahara_hp={halfpad}'
    processed = kuwahara_response(imageMatrix,hp=2)
    Test.saveOutput(imageMatrix,processed,figName=testName)