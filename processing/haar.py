import numpy as np
import matplotlib.pyplot as plt


def time_eval(t):
    if 0 <= t < 1 / 2:
        return 1
    elif 1 / 2 < t < 1:
        return -1
    else:
        return 0


def mother_wave():
    t = np.linspace(0, 1, 200)
    return t, np.array([time_eval(i) for i in t])


def indicatrices(min, sup, t):
    if min <= t < sup:
        return 1
    return 0


def haar_k_l(k, l):
    if k < 0:
        raise ValueError('k must be >0')
    if l <= 1 or l >= 2**k:
        raise ValueError('l must be in [1,2**k]')
    t = np.linspace(0, 1, 500)
    min1 = float(2 * l - 2) / float(2**(k + 1))
    sup1 = float(2 * l - 1) / float(2**(k + 1))
    func1 = np.array([indicatrices(min1, sup1, ti) for ti in t])

    min2 = float(2 * l - 1) / float(2**(k + 1))
    sup2 = float(2 * l) / float(2**(k + 1))
    func2 = np.array([indicatrices(min2, sup2, ti) for ti in t])

    return func1 - func2
