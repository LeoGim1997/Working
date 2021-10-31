import numpy as np
import matplotlib.pyplot as plt
from math import pi
"""
Objectifs :
"""

def f(x):
    return np.cos(x)-x**3

def derivate_f(x):
    return -np.sin(x)-2*x**2

def Newton_method(x0,epsilon=10**(-3),max_iter=1000):
    xn= x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = derivate_f(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None

def visu():
    x = np.linspace(-4*pi,4*pi,1000)
    y = f(x)
    plt.figure()
    plt.grid()
    plt.plot(x,y)
    plt.show()

if __name__=="__main__":
    x_init = -0.5
    visu()
    x = Newton_method(x_init,epsilon=10**(-3))
    print(f"x={x}")


