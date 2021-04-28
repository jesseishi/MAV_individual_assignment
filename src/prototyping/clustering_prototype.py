# Imports.
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
# Clustering from: https://machinelearningmastery.com/clustering-algorithms-with-python/
from sklearn.datasets import make_classification
from sklearn.cluster import Birch, DBSCAN
from itertools import permutations, combinations

# Load an image.
data_folder = os.path.abspath(os.path.join(os.curdir, '../../..', 'WashingtonOBRace'))
img = cv2.imread(os.path.join(data_folder, 'img_132.png'), 0)

# Initiate ORB detector.
orb = cv2.ORB_create()

# Find the keypoints with ORB.
kps = orb.detect(img, None)

# Compute the descriptors with ORB.
kps, des = orb.compute(img, kps)

# Get the x,y coordinates.
X = np.array([[*kp.pt] for kp in kps])

# # Define the model.
# model = Birch(threshold=0.01, n_clusters=10)
#
# # Fit the model.
# model.fit(X)
#
# # Assign a cluster to each example.
# yhat = model.predict(X)
# Define the model - DBSCAN.
# X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
model = DBSCAN(eps=10, min_samples=10)
y_hat = model.fit_predict(X)


# Get the centroids of all clusters.
def get_cluster_centroid(i, yhat, X):
    row_i = np.where(yhat == i)
    return np.average(X[row_i, 0]), np.average(X[row_i, 1])


clusters = np.unique(y_hat)
clusters_coordinates = np.zeros((len(clusters), 2))
for cluster in clusters:
    clusters_coordinates[cluster, :] = get_cluster_centroid(cluster, y_hat, X)


# Does a set of 4 coordinates form a nice rectangle?
def calc_rectangle_fit(v1, v2, v3, v4):

    # A simple test would be to check if the four shortest distances between the vertices are the same length.
    lengths_between_vertices = []

    # Loop through all combinations and get the distance between them.
    for a, b in combinations([v1, v2, v3, v4], 2):
        lengths_between_vertices.append(np.linalg.norm(b - a))

    # The score is the inverse of the standard deviation of the lowest four lengths.
    lengths_between_vertices.sort()
    return 1 / np.std(lengths_between_vertices[:4])


# Loop through all possible orders of clusters and check if they form a rectangle.
clusters_fit = []
for i, (a, b, c, d) in enumerate(combinations(clusters_coordinates, 4)):

    clusters_fit.append([calc_rectangle_fit(a, b, c, d), [a, b, c, d]])

# Plotting.
# Plot the original image with keypoints.
img2 = cv2.drawKeypoints(img, kps, outImage=None, color=(0, 255, 0), flags=0)
plt.imshow(img2)

# Create scatter plot for samples from each cluster.
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = np.where(y_hat == cluster)

    # Create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1])

    # Plot the cluster centroids.
    # plt.plot(*clusters_coordinates[cluster, :], 'X', markersize=10)

# Plot our best idea of the rectangle.
clusters_fit.sort()
rectangle_coordinates = np.asarray(clusters_fit[-1][1])
plt.plot(rectangle_coordinates[:, 0], rectangle_coordinates[:, 1], 'kX', markersize=20)

# Show it.
plt.show()
