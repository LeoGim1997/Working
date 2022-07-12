import pytest
from num.minmax import reflectmode

l1 = [1, 2, 3]
l2 = [1, 2, 5, 5, 5, 5]


@pytest.mark.parametrize('input,size', [(l1, 3)])
def test_reflectmode1(input, size):
    a = reflectmode(input, size)
    assert isinstance(a, list)


@pytest.mark.parametrize('input,size', [(l1, 3), (l2, 4)])
def test_reflectmode2(input, size):
    a = reflectmode(input, size)
    assert len(a) == 2 * size + len(input)


@pytest.mark.parametrize('input,size', [([1, 2, 4], 3)])
def test_reflectmodeLeft(input, size):
    a = reflectmode(input, size)
    assert a[0:size] == [4, 2, 1]


@pytest.mark.parametrize('input,size', [([1, 2, 4], 3)])
def test_reflectmodeRight(input, size):
    a = reflectmode(input, size)
    assert a[len(a) - size:] == [4, 2, 1]


@pytest.mark.parametrize('input,size', [([5, 2, 3], 3)])
def test_reflectmodefull(input, size):
    a = reflectmode(input, size)
    assert a == [3, 2, 5, 5, 2, 3, 3, 2, 5]


@pytest.mark.parametrize('input,size', [([5, 2, 3], 12)])
def test_reflectexception(input, size):
    with pytest.raises(ValueError, match=f'window of {size} larger than total input length.'):
        reflectmode(input, size)
