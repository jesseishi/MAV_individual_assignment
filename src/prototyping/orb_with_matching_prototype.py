# import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# img1 = cv2.imread('this_is_what_a_gate_looks_like3.png', 0)          # queryImage
# img1 = cv2.imread('this_is_what_a_corner_looks_like.png', 0)          # queryImage
# img1 = cv2.imread('images/blokje2.png', 0)          # queryImage
img1 = cv2.imread('img_315_flat.png', 0)
# img2 = cv2.imread('box_in_scene.png',0) # trainImage

data_folder = os.path.abspath(os.path.join(os.curdir, '../../..', 'WashingtonOBRace'))
img2 = cv2.imread(os.path.join(data_folder, 'img_57.png'), 0)

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50], outImg=None, flags=2, matchColor=(0, 0, 255))

# # Draw all matches.
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches, outImg=None, flags=2)

plt.imshow(img3)
plt.xticks([])
plt.yticks([])
plt.show()
