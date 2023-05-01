import pytest
from dithering import snakePath,diagonal



@pytest.mark.parametrize('n,m, gen',[(
        2,2,((0,0),(0,1),(1,-1),(1,-2))
)])
def test_generator(n,m,gen):
    a = tuple(snakePath(n,m))
    assert a == gen
