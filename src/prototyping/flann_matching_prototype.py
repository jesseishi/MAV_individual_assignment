import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# img1 = cv2.imread('box.png',0)          # queryImage
# img1 = cv2.imread('this_is_what_a_corner_looks_like.png', 0)          # queryImage
img1 = cv2.imread('images/this_is_what_a_gate_looks_like3.png', 0)          # queryImage
# img1 = cv2.imread('blokje2.png', 0)          # queryImage
# img1 = cv2.imread('img_315_flat.png', 0)  # For some fucking reason doesn't work.

data_folder = os.path.abspath(os.path.join(os.curdir, '../../..', 'WashingtonOBRace'))
img2 = cv2.imread(os.path.join(data_folder, 'img_28.png'), 0)

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 1  # TODO: Not sure whether this is the right value.
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
index_params = dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6,# 6, # 12
                    key_size = 12, # 12,     # 20
                    multi_probe_level = 1,) #1) #2
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)


# Convert to float 32 ndarrays.
# https://stackoverflow.com/questions/12508934/error-using-knnmatch-with-opencvpython
des1 = np.asarray(des1, dtype=np.float32)
des2 = np.asarray(des2, dtype=np.float32)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:  # Used to be 0.7.
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()
