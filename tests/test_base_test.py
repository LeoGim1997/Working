import pytest
from base_test import Test
from im import MyImage


@pytest.mark.Base
@pytest.mark.parametrize(
    "im1,cmap",
    [("lena", "gray")],
)
def test_fig_creation(im1, im2, cmap):
    figName,ext = "LenaNoProcessing","png"
    lenamatrix = MyImage(im1).get_matrix()
    Test.saveOutput(lenamatrix,lenamatrix,cmap=cmap,figName=figName)
    assert (Test.outputPath / f"{figName}.{ext}").exists()
