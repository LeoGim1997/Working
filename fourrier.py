# Comparaison de plusieurs algorithmes pour evaluer la TFD d'un signal discret
import numpy as np
import matplotlib.pyplot as plt
from pandas import array
from PIL import Image
from im import normalize
# Algo 1 : Naif

def compute_DFT(x : np.array ) -> np.array:
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X

def exp_coeff(u, m, M, v, n, N):
    coeff = -2j*np.pi*(((u*m)/M)+((v*n)/N))
    return np.exp(coeff)


def compute_DFT_image_coeff(image : np.array , u,v) -> np.array:
    N,M = np.shape(image)
    dft = np.zeros((N,M),dtype=complex)
    for m in range(N-1):
        for n in range(M-1):
            a = exp_coeff(u,m,M,v,n,N)
            dft[m,n] = complex(image[m,n]*a)
    return np.sum(dft)


def compute_DFT_image( image : np.array ) -> np.array:
    N,M = np.shape(image)
    dft = np.zeros((N,M),dtype=complex)
    for u in range(N-1):
        for v in range(M-1):
            dft[u,v] = compute_DFT_image_coeff(image,u,v)
    return (1/N*M)*dft


def cosinus_generation( freq , amplitude,max=1):
    """
    Generate an 1-D Cosinus array with the given amplitude
    and frequency
    """
    step =1/(2*freq)
    N = int(max/step)
    t = np.linspace(0,max,N)
    return t,amplitude*np.cos(freq*t)
