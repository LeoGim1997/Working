import pytest
import os
from pathlib import Path
import sys
import numpy as np
sys.path.append(Path(__file__).parents[1].as_posix())
from processing.noise import gaussian_noise
from confest import lena_standard


@pytest.mark.Noise
def testGaussianNoise(lena_standard):
    noise = gaussian_noise(img=lena_standard, mean=0, sigma=1)
    assert np.shape(noise) == np.shape(lena_standard)
