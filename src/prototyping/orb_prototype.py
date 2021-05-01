import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# img = cv2.imread('simple.jpg',0)
data_folder = os.path.abspath(os.path.join(os.curdir, '../../..', 'WashingtonOBRace'))
img = cv2.imread(os.path.join(data_folder, 'img_57.png'), 0)

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img, None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location, not size and orientation
img2 = cv2.drawKeypoints(img, kp, outImage=None, color=(0, 0, 255), flags=0)
plt.imshow(img2)
plt.xticks([])
plt.yticks([])
plt.show()
