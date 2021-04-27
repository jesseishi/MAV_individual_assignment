# Imports.
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
# import pandas as pd  # TODO: Add to requirements.


# Load a picture.
imgname = 'img_8.png'
data_folder = os.path.abspath(os.path.join(os.curdir, '../../..', 'WashingtonOBRace'))
img = cv2.imread(os.path.join(data_folder, imgname))

# And the mask.
maskname = 'mask_8.png'
img = cv2.imread(os.path.join(data_folder, maskname), 0)

# Load the coordinates.
# TODO: just copied from now.
coordinates1 = [91,134,164,135,167,197,93,203]
coordinates2 = [4,147,47,149,49,185,6,186]

# Plot the coordinates in the image.
plt.imshow(img)
# plt.imshow(mask)
plt.plot(coordinates1[::2], coordinates1[1::2], 'kX')
plt.plot(coordinates2[::2], coordinates2[1::2], 'kX')
plt.show()
