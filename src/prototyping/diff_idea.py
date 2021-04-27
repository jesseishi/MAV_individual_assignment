# If you differentiate enough times, the clear checkerboard thing is the only thing left.

# Imports.
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

data_folder = os.path.abspath(os.path.join(os.curdir, '../../..', 'WashingtonOBRace'))
img = cv2.imread(os.path.join(data_folder, 'img_182.png'), 0)

N = 4
fig, ax = plt.subplots(1, N)

for i in range(N):
    ax[i].imshow(img, 'gray')
    img = np.diff(img)

plt.show()