import pytest
from num.minmax import reflectmode


@pytest.mark.parametrize('input,size', [([1, 2, 3], 3)])
def test_reflectmode(input, size):
    a = reflectmode(input, size)
    assert isinstance(a, list)
