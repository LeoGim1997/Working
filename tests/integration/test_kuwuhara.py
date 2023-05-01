import pytest
from pathlib import Path
import sys
from base_test import Test
from processing.im import MyImage

from processing.kuwahara import kuwahara_response

def generateImage(imgName : str , **kwargs) -> None:
    imageMatrix = MyImage(imgName).get_matrix()
    testName = f'{imgName}_kuwahara' + '_'.join((f'{k}={v}' for k,v in kwargs.items()))
    processed = kuwahara_response(imageMatrix,**kwargs)
    Test.saveOutput(imageMatrix,processed,figName=testName)


@pytest.mark.integration
@pytest.mark.parametrize('image', ['lena'])
def test_kuwahara_response(image):
    generateImage(image)


@pytest.mark.integration
@pytest.mark.parametrize('image',['lena_couleur'])
def test_kuwahara_couleur(image):
    generateImage(image)

@pytest.mark.integration
@pytest.mark.parametrize('halfpad',(2,4,5,9,15))
def test_multiple_halpad(halfpad):
    for image in ('david','lena_couleur'):
        generateImage(image,hp=halfpad)
