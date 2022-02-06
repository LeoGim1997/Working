import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from im import normalize
img = mpimg.imread('/Users/leogimenez/Desktop/git_depo_local/Working/image/lena_gray.gif')

img_norm = normalize(img)
plt.imshow(img_norm)
plt.show()