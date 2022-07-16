import numpy as np
from collections.abc import Iterable
from inspect import signature


def maxAlongAxis(input, size, mode='reflect', cval=0.0):
    if not isinstance(input, Iterable):
        raise ValueError('input value is not Iterable.')
    modifiedType = type(input)
    if not isinstance(modifiedType, (list, tuple)):
        modifiedType = np.array
    input = list(input)
    mapDict = {
        'reflect': reflectmode,
        'constant': constantmode,
        'nearest': nearestmode,
    }
    args = locals()
    try:
        mapDict[mode]
    except Exception as e:
        raise ValueError(f'Incorrect mode =  {mode}')
    postP = mapDict[mode](**args)
    c = int(size / 2)
    at = [max(postP[i - c:i + c + 1]) for i in range(size, len(postP) - size)]

    return modifiedType(at)


def reflectmode(**args) -> Iterable:
    input, size = args.get('input'), args.get('size')
    if size > len(input):
        raise ValueError(f'window of {size} larger than total input length.')
    g = input[0:size]
    d = input[len(input) - size:]
    g = g[::-1]
    d = d[::-1]
    return g + input + d


def constantmode(*arg, **args) -> Iterable:
    input, size = args.get('input'), args.get('size')
    cval = args.get('cval')
    if size > len(input):
        raise ValueError(f'window of {size} larger than total input length.')
    s = len(input[0:size])
    g = [cval for _ in range(s)]
    return g + input + g


def nearestmode(**args) -> Iterable:
    input, size = args.get('input'), args.get('size')
    g = [input[0] for _ in range(size)]
    return g + input + g


def maxAlongImg(img, axis=0, size=3, mode='reflect', cval=0.0):
    max_img = np.copy(img)
    if axis == 1:
        max_img = max_img.T
    for i_row, row in enumerate(max_img):
        row = maxAlongAxis(row, size, mode, cval)
        max_img[i_row, :] = row
    return max_img
