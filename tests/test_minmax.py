import pytest
from num.minmax import *

l1 = [1, 2, 3]
l2 = [1, 2, 5, 5, 5, 5]


@pytest.mark.parametrize('args', [{'input': l1, 'size': 3}])
def test_reflectmode1(args):
    a = reflectmode(**args)
    assert isinstance(a, list)


@pytest.mark.parametrize('args', [{'input': l1, 'size': 3}, {'input': l2, 'size': 4}])
def test_reflectmode2(args):
    a = reflectmode(**args)
    assert len(a) == 2 * args['size'] + len(args['input'])


@pytest.mark.parametrize('args', [{'input': [1, 2, 4], 'size': 3}])
def test_reflectmodeLeft(args):
    a = reflectmode(**args)
    size = args['size']
    assert a[0:size] == [4, 2, 1]


@pytest.mark.parametrize('args', [{'input': [1, 2, 4], 'size': 3}])
def test_reflectmodeRight(args):
    a = reflectmode(**args)
    size = args['size']
    assert a[len(a) - size:] == [4, 2, 1]


@pytest.mark.parametrize('args', [{'input': [5, 2, 3], 'size': 3}])
def test_reflectmodefull(args):
    a = reflectmode(**args)
    assert a == [3, 2, 5, 5, 2, 3, 3, 2, 5]


@pytest.mark.parametrize('args', [{'input': [5, 2, 3], 'size': 12}])
def test_reflectexception(args):
    input, size = args.get('input'), args.get('size')
    with pytest.raises(ValueError, match=f'window of {size} larger than total input length.'):
        reflectmode(**args)


@pytest.mark.parametrize('args', [{'input': l1, 'size': 3, 'cval': 1}])
def test_constantmode1(args):
    a = constantmode(**args)
    assert isinstance(a, Iterable)


@pytest.mark.parametrize('args', [{'input': l1, 'size': 3, 'cval': 1}])
def test_constantmode2(args):
    a = constantmode(**args)
    cval, size = args.get('cval'), args.get('size')
    assert all([c == cval for c in a[0:size]])


@pytest.mark.parametrize('args', [{'input': l1, 'size': 3, 'cval': 8}])
def test_constantmode2(args):
    a = constantmode(**args)
    assert a == [8, 8, 8, 1, 2, 3, 8, 8, 8]


@pytest.mark.parametrize('args', [{'input': l1, 'size': 3}])
def test_nearestmode(args):
    a = nearestmode(**args)
    assert isinstance(a, Iterable)


@pytest.mark.parametrize('args', [{'input': l1, 'size': 3}])
def test_nearestmode(args):
    a = nearestmode(**args)
    assert isinstance(a, Iterable)
