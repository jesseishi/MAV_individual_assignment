import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# img1 = cv2.imread('box.png',0)          # queryImage
# img1 = cv2.imread('this_is_what_a_corner_looks_like.png')          # queryImage
img1 = cv2.imread('images/this_is_what_a_gate_looks_like2.png')          # queryImage
# img1 = cv2.imread('img_315_flat.png')          # queryImage

data_folder = os.path.abspath(os.path.join(os.curdir, '../../..', 'WashingtonOBRace'))
img2 = cv2.imread(os.path.join(data_folder, 'img_182.png'))  # TODO: look what kwarg 0 does, I think greyscaling.

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
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)


# Convert to float 32 ndarrays.
# https://stackoverflow.com/questions/12508934/error-using-knnmatch-with-opencvpython
des1 = np.asarray(des1, dtype=np.float32)
des2 = np.asarray(des2, dtype=np.float32)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper.
# https://docs.opencv.org/3.1.0/d1/de0/tutorial_py_feature_homography.html
good_matches = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.95*n.distance:  # Used to be 0.7.
        good_matches.append(m)

MIN_MATCH_COUNT = 4
if len(good_matches)>=MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w,d = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(good_matches),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,None,**draw_params)

plt.imshow(img3, 'gray')
plt.show()
