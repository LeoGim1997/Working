import pytest
import numpy as np
import sys
from base_test import Test
sys.path.append(Test.processingPath.resolve().as_posix()) #type : ignore
from dithering import snakePath



@pytest.mark.parametrize('n,m, gen',[(
        2,2,((0,1),(-1,2))
)])
def test_generator(n,m,gen):
    a = tuple(snakePath(n,m))
    assert a == gen