import numpy as np
from collections.abc import Iterable


def maxAlongAxis(input, size, mode='reflect', cval=0.0, origin=0, keepType=True):
    if not isinstance(input, Iterable):
        input = list(input)
        modifiedType = type(input)
    mapDict = {
        'reflect': reflectmode,
        'constant': constantmode,
        'nearest': nearestmode,
    }
    postProcess = mapDict.get(mode)(input, size)
    # TODO : Implement those 3 functions


def reflectmode(input: list, size: int) -> Iterable:
    if size > len(input):
        raise ValueError(f'window of {size} larger than total input length.')
    g = input[0:size]
    d = input[len(input) - size:]
    g = g[::-1]
    d = d[::-1]
    return g + input + d


def constantmode(input: list, size: int, cval: float) -> Iterable:
    pass


def nearestmode(input: list, size: int) -> Iterable:
    pass
