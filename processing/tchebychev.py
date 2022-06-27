import numpy as np
import matplotlib.pyplot as plt


def T(d, x):
    """Eval polynom of degree d at point x.

    Parameters:
    -----------
        d: int
            degree of the require polynom.
        x: float
            point of the evaluation.
    Returns
    -------
        Tx: float
            evaluate point.
    """
    if d < 0:
        raise ValueError(f'Degree cannot be negative.')
    if d == 0:
        return 1
    if d == 1:
        return x
    else:
        return 2 * x * T(d - 1, x) - T(d - 2, x)


def showTemplate():
    x = np.linspace(-1, 1, 100)
    plt.figure()
    legend = []
    for d in range(6):
        Tx = [T(d, e) for e in x]
        plt.plot(x, Tx)
        legend += [f'Order d={d}']
    plt.legend(legend)
    plt.grid()
    plt.show()


def tchebychevOrdre1(w, n, wc=1, eps=1):
    elem = (eps**2) * T(n, w / wc)**2
    return 1 / np.sqrt(1 + elem)


def tchebychevOrdre2(w, n, wc=1, eps=1) -> float:
    num = eps * T(n, wc / w)
    den = eps**2 * (T(n, wc / w)**2)
    return num / np.sqrt(1 + den)


def get_response(*arg, order=1):
    params = arg[1:][0]
    wvec = arg[0]
    if np.shape(arg[0])[0] == 0:
        print(f'Wrong input for w vectors')
        return False
    if order == 1:
        filter = tchebychevOrdre1
    if order == 2:
        filter = tchebychevOrdre2
    response = [filter(w, *params) for w in wvec]
    return np.array(response)
