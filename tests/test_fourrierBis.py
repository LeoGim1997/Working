import pytest
import sys
from pathlib import Path
import numpy as np
sys.path.append(Path(__file__).parents[1].as_posix())
from processing.fourrierBis import fft, fft_v

short_x = np.linspace(1, 100, 1)
normal_x = np.linspace(1, 100, 2)


def dirac(N: int = 64):
    d = np.zeros((N, 1))
    d[int(N // 2)] = 1
    return d


@pytest.mark.parametrize('input', [short_x])
def test_exception(input):
    msg = 'wrong shape 1 for x. Must be a power of 2.'
    with pytest.raises(ValueError, match=msg) as exc_info:
        fft(input)


@pytest.mark.parametrize('input', [normal_x])
def test_fourrier(input):
    tf = fft(input)
    assert isinstance(tf, (np.ndarray, np.generic))


@pytest.mark.parametrize('input', [normal_x])
def test_fourrier_len(input):
    tf = fft(input)
    assert tf.shape == input.shape


@pytest.mark.parametrize('func,N', [(dirac, 64)])
def test_fft_dirac(func, N):
    tf = fft(func(N))
    assert tf.shape[0] == N


@pytest.mark.parametrize('func,N', [(dirac, 64)])
def test_fft_dtype(func, N):
    tf = fft(func(N))
    assert tf.dtype == 'complex128'


@pytest.mark.parametrize('func,N', [(dirac, 32)])
def test_fft_firstvalue(func, N):
    tf = fft(func(N))
    # check if first value are 1 in module
    assert tf[0] == 1


@pytest.mark.parametrize('func,N', [(dirac, 32)])
def test_fft_allvalues(func, N):
    tf = fft(func(N))
    assert all([np.abs(e) == pytest.approx(1, 10e-3) for e in tf])


@pytest.mark.parametrize('type',
                         [[1, 2, 3, 4],
                          (1, 2, 3, 4),
                          np.linspace(0, 3, 4)])
def test_fft_dtype_input(type):
    tf = fft(type)
    assert isinstance(tf, (np.generic, np.ndarray))
    assert tf.shape[0] == 4
