import numpy as np

# Methode 1
a = [0, 11, 4, 3, 8, 5, 6, 1, 7, 3, 10]

k = 3
g = a[0:k]
d = a[len(a) - k:]
g = g[::-1]
d = g[::-1]
b = g + a + d

c = int(k / 2)
at = [max(b[i - c:i + c + 1]) for i in range(k, len(b) - k)]
at = np.array(at)
print(f'methode1 = {at}')

from scipy.ndimage import maximum_filter1d
print(f'methode2 = {maximum_filter1d(a, size=k)}')
