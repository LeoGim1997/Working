import numpy as np
from genericImage import straight_line
import matplotlib.pyplot as plt
maxTheta = 90
minTheta = -maxTheta
img = straight_line()
n, m = img.shape
rshape = max(n, m)
theta = np.linspace(0, 180)
accumulator = np.zeros((rshape + 2, theta.shape[0] + 2))

for i in range(n):
    for j in range(m):
        if img[i, j] == 0:
            continue
        for c, a in enumerate(theta):
            r = i * np.cos(a) + j * np.sin(a)
            if np.abs(r) > img.shape[0]:
                r = img.shape[0] if r > 0 else -1 * img.shape[0]
            accumulator[int(r) - 1, c] += 1

h = accumulator

plt.figure()
plt.imshow(np.log(1 + h),
           extent=[theta[0], theta[-1],
                   -150, 150],
           cmap=plt.cm.gray, aspect=1 / 1.5)
plt.show()
