import pytest
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from processing.noise import gaussian_noise  # noqa : E402


# @pytest.mark.Noise
# @pytest.mark.parametrize("args", [{'mean': 0, 'sigma': 1}])
# def testGaussianNoise(args):
#     gaussian_noise(args)
