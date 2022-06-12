import numpy as np
from typing import Iterable, Tuple


def rect(x: Iterable, y: Iterable, bound: Tuple[float, float] = None) -> float:
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
        raise ValueError(f'x and y are not the same size.')
    if bound is not None:
        for v in bound:
            if (x[0] > v) or (x[-1] < v):
                raise ValueError(f'function is not defined on' +
                                 'the specified bound.')
    n = len(x)
    integeral = 0
    for i in range(n - 1):
        integeral += y[i] * (x[i + 1] - x[i])
    return integeral


def trap(x: Iterable, y: Iterable, bound: Tuple[float, float] = None) -> float:
    """Rectangle mehod for integral.

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
