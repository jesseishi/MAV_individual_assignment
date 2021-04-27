# Imports.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Copied from: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html
# The folder with this repo should be in the same folder as the data folder 'WashingtonOBRace'.
data_folder = os.path.abspath(os.path.join(os.curdir, '../../..', 'WashingtonOBRace'))
img = cv2.imread(os.path.join(data_folder, 'img_182.png'), 0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, outImage=None, color=(255,0,0))

# Print all default params
# print("Threshold: ", fast.getInt('threshold'))
# print("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
# print("neighborhood: ", fast.getInt('type'))
print("Total Keypoints with nonmaxSuppression: ", len(kp))

# cv2.imwrite('fast_true.png',img2)
plt.imshow(img2)
plt.show()

# # Disable nonmaxSuppression
# fast.setBool('nonmaxSuppression',0)
# kp = fast.detect(img,None)
#
# print("Total Keypoints without nonmaxSuppression: ", len(kp))
#
# img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))
#
# cv2.imwrite('fast_false.png',img3)
