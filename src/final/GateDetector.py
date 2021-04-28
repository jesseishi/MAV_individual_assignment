# Imports.
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from itertools import combinations


# The final class. Takes an image and tries to find the coordinates of the corners of the gate.
# TODO: Some explanation on how it works.
class GateDetector:
    def __init__(self, detector_params, cluster_model_params):

        # We use the ORB detector.
        self.orb = cv2.ORB_create(**detector_params)

        # We use the DBSCAN clustering algorithm.
        self.dbscan = DBSCAN(**cluster_model_params)

    # Main function that returns the coordinates of the corners of the gate in the image.
    def detect_gate(self, im):

        # Get the coordinates of the keypoints.
        X = self._get_keypoint_coordinates(im)

        # Cluster the features using a clustering algorithm and find the centroids of these clusters.
        cluster_coordinates = self._get_cluster_coordinates(X)

        # Find the best fitting rectangle out of all combinations of cluster coordinates.
        best_fit_coordinates = self._fit_rectangle(cluster_coordinates)

        # Return the coordinates of the four corners.
        # Note that they may be in random order.
        return best_fit_coordinates

    def _get_keypoint_coordinates(self, im):
        # Detect features.
        kps = self.orb.detect(im, None)
        kps, des = self.orb.compute(im, kps)  # TODO: Does this line change the kps?

        # Get the coordinates out of the keypoint objects and return them.
        return np.array([[*kp.pt] for kp in kps])

    def _get_cluster_coordinates(self, X):

        # Find the different clusters.
        y_hat = self.dbscan.fit_predict(X)

        # For each unique cluster, get the coordinates of the centroid.
        clusters = np.unique(y_hat)
        clusters_coordinates = np.zeros((len(clusters), 2))

        for cluster in clusters:

            # Find which rows of X belong to this cluster and take the average of their coordinates.
            row_i = np.where(y_hat == cluster)
            clusters_coordinates[cluster, :] = np.average(X[row_i, 0]), np.average(X[row_i, 1])

        return clusters_coordinates

    def _fit_rectangle(self, coordinates):

        # Loop through all possible orders of clusters and check if they form a rectangle.
        # We store both the score and the four coordinates that form this rectangle.
        clusters_fit = []
        for i, (a, b, c, d) in enumerate(combinations(coordinates, 4)):
            clusters_fit.append([self._calc_rectangle_fit(a, b, c, d), [a, b, c, d]])

        # Sort the fits by score.
        clusters_fit.sort()

        # Return the four best coordinates.
        return np.asarray(clusters_fit[-1][1])

    def _calc_rectangle_fit(self, v1, v2, v3, v4):
        # TODO: make a better fitting function. Maybe taking into account perspective, so if it looks like a trapezoid
        #  that's also good... Or something like that.

        # A simple test would be to check if the four shortest distances between the vertices are the same length.
        lengths_between_vertices = []

        # Loop through all combinations and get the distance between them.
        for a, b in combinations([v1, v2, v3, v4], 2):
            lengths_between_vertices.append(np.linalg.norm(b - a))

        # The score is the inverse of the standard deviation of the lowest four lengths.
        lengths_between_vertices.sort()
        return 1 / np.std(lengths_between_vertices[:4])
