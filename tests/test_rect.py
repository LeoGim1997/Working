import pytest
import sys
from pathlib import Path
import numpy as np
sys.path.append(Path(__file__).parents[1].as_posix())
from num.rect import rectIntegral, trap


def vector(xo, xn, n):
    x = np.linspace(xo, xn, n)
    return x


def line(xo, xn, n):
    """
    Returns a line for UT.
    """
    x = np.linspace(xo, xn, n)
    y = np.ones(n)
    return x, y


@pytest.mark.Num
@pytest.mark.parametrize('xparams', [(0, 1, 100)])
@pytest.mark.parametrize('yparams', [(0, 1, 100)])
def test_implement(xparams, yparams):
    x = vector(*xparams)
    y = vector(*yparams)
    result = rectIntegral(x, y)
    assert isinstance(result, np.float64)


@pytest.mark.Num
@pytest.mark.parametrize('xparams', [(0, 1, 15)])
@pytest.mark.parametrize('yparams', [(0, 1, 100)])
def test_different_len(xparams, yparams):
    x = vector(*xparams)
    y = vector(*yparams)
    with pytest.raises(ValueError,
                       match='x and y are not the same size.'):
        rectIntegral(x, y)


@pytest.mark.Num
@pytest.mark.parametrize('params', [(0, 1, 10)])
@pytest.mark.parametrize('bound', [(8, 9)])
def test_wrong_bound(params, bound):
    x = vector(*params)
    with pytest.raises(ValueError,
                       match='function is not defined ' +
                       'on the specified bound.'):
        rectIntegral(x, x, bound=bound)


@pytest.mark.Num
@pytest.mark.parametrize('params', [(0, 5, 100)])
def test_line(params):
    x, y = line(*params)
    result = rectIntegral(x, y)
    assert result == 5.0
