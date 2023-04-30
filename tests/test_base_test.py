import pytest
from base_test import Test
from im import MyImage


@pytest.mark.Base
@pytest.mark.parametrize(
    "im1,im2,cmap",
    [(MyImage("lena").get_matrix(), MyImage("lena").get_matrix(), "gray")],
)
def test_fig_creation(im1, im2, cmap):
    figName = "LenaNoProcessing"
    print(im1,im2)
    Test.saveOutput(im1, im2)
    assert (Test.outputPath / figName).exists()
