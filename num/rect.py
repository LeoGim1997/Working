import numpy as np
from typing import Iterable, Tuple


def truncate(x: Iterable, bound=None):
    """Truncate function
    Allows the truncation on specified bound.

    Parameters
    ----------
    x : Iterable
        input value to truncate.
    bound: Optional-tuple : Bound for the truncation.

    Returns
    -------
    xf : array_like
        truncated value of the input x.
    """
    if bound is None:
        return x
    if len(bound) != 2:
        raise ValueError('Both min&max must be specified.')
    if bound[0] > bound[1]:
        raise ValueError(f'Wrong order for bound spec {bound[0]} >{bound[1]}.')
    mi, ma = bound
    xf = [e for e in x if e >= mi or e <= ma]
    return np.array(xf)


def rectIntegral(x: Iterable, y: Iterable, bound: Tuple[float, float] = None) -> float:
    """Rectangle method for integral.
    Compute the integral approximation of a function using
    rectangle method.

    Parameters
    ----------
    x : Iterable
        discrete value of the x axis.
    y : Iterable
        discrete value of the y axis.

    bound : tuple , optional
            bound where to compute the integral.

    Returns
    --------
    integral: float
        Value of the computed integral.

    Raises
    ------
    ValueError
        If bound is not inside the function value.
        If `len(x)` != `len(y)`

    Notes
    -----
        This function uses the standard rectangle
        method approach to compute the integral.
    """
    if len(x) != len(y):
        raise ValueError('x and y are not the same size.')
    if bound is not None:
        for v in bound:
            if (x[0] > v) or (x[-1] < v):
                raise ValueError('function is not defined on ' +
                                 'the specified bound.')
    n = len(x)
    integeral = 0
    for i in range(n - 1):
        integeral += y[i] * (x[i + 1] - x[i])
    return integeral


def trap(x: Iterable, y: Iterable, bound: Tuple[float, float] = None) -> float:
    """Trapeze mehod for integral.

    Compute the integral approximation of a function using
    trapeze method.

    Parameters
    ----------
    x : Iterable
        discrete value of the x axis.
    y : Iterable
        discrete value of the y axis.

    bound : tuple , optional
            bound where to compute the integral.

    Returns
    --------
    integral: float
        Value of the computed integral.

    Raises
    ------
    ValueError
        If bound is not inside the function value.
        If `len(x)` != `len(y)`

    Notes
    -----
        This function uses the standard trapeze
        method approach to compute the integral.
    """
    if len(x) != len(y):
        raise ValueError(f'x and y are not the same size.')
    if bound is not None:
        for v in bound:
            if (x[0] > v) or (x[-1] < v):
                raise ValueError(f'function is not defined on' +
                                 'the specified bound.')
    n = len(x)
    integeral = 0
    for i in range(n - 1):
        integeral += (y[i + 1] + y[i]) * ((x[i + 1] - x[i])) / 2
    return integeral
