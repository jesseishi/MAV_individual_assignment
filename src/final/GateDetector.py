# Imports.
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from itertools import combinations


# The final class. Takes an image and tries to find the coordinates of the corners of the gate.
# TODO: Some explanation on how it works.
class GateDetector:
    def __init__(self, detector_params, cluster_model_params, gate_width_over_length_ratio=0.275):

        # We use the ORB detector.
        self.orb = cv2.ORB_create(**detector_params)

        # We use the DBSCAN clustering algorithm.
        self.dbscan = DBSCAN(**cluster_model_params)

        # The only other parameter we need is to estimate the mask from the coordinates.
        self.gate_width_over_length_ratio = gate_width_over_length_ratio

    # Main function that returns the coordinates of the corners of the gate in the image.
    def detect_gate(self, im, return_mask=False):

        # Get the coordinates of the keypoints.
        self.X = self._get_keypoint_coordinates(im)

        # Cluster the features using a clustering algorithm and find the centroids of these clusters.
        cluster_coordinates = self._get_cluster_coordinates(self.X)

        # Find the best fitting rectangle out of all combinations of cluster coordinates.
        best_score, best_coordinates = self._fit_rectangle(cluster_coordinates, np.shape(im))

        if return_mask:
            # Estimate the mask of the gate.
            mask_hat = self._estimate_mask(best_score, best_coordinates, np.shape(im))

            # Return the coordinates of the four corners.
            # Note that they may be in random order.
            return best_coordinates, mask_hat
        else:
            return best_coordinates

    def _get_keypoint_coordinates(self, im):
        # Detect features.
        kps = self.orb.detect(im, None)
        kps, des = self.orb.compute(im, kps)  # TODO: Does this line change the kps?

        # Get the coordinates out of the keypoint objects and return them.
        return np.array([[*kp.pt] for kp in kps])

    def _get_cluster_coordinates(self, X):

        # Find the different clusters.
        self.y_hat = self.dbscan.fit_predict(X)

        # For each unique cluster, get the coordinates of the centroid.
        clusters = np.unique(self.y_hat)
        clusters_coordinates = np.zeros((len(clusters), 2))

        for cluster in clusters:

            # Find which rows of X belong to this cluster and take the average of their coordinates.
            row_i = np.where(self.y_hat == cluster)
            clusters_coordinates[cluster, :] = np.average(X[row_i, 0]), np.average(X[row_i, 1])

        return clusters_coordinates

    def _fit_rectangle(self, coordinates, shape):

        # Loop through all possible orders of clusters and check if they form a rectangle.
        # We store both the score and the four coordinates that form this rectangle.
        clusters_fit = []
        for i, (a, b, c, d) in enumerate(combinations(coordinates, 4)):
            clusters_fit.append([self._calc_rectangle_fit(a, b, c, d), [a, b, c, d]])

        # Sort the fits by score.
        clusters_fit.sort(key=lambda x: x[0])

        # We now want to sort the coordinates by going from the top-left and the clockwise.
        # We can do this by calculating the norm to the corners of the image.
        best_coordinates = clusters_fit[-1][1]
        tl_arg = np.array([np.linalg.norm(np.array([0, 0]) - coord) for coord in best_coordinates]).argmin()
        tr_arg = np.array([np.linalg.norm(np.array([shape[0], 0]) - coord) for coord in best_coordinates]).argmin()
        br_arg = np.array([np.linalg.norm(np.array([shape[0], shape[1]]) - coord) for coord in best_coordinates]).argmin()
        bl_arg = np.array([np.linalg.norm(np.array([0, shape[1]]) - coord) for coord in best_coordinates]).argmin()

        sorted_coordinates = np.array([best_coordinates[tl_arg], best_coordinates[tr_arg],
                                       best_coordinates[br_arg], best_coordinates[bl_arg]])

        # Return the highest score and the four best coordinates.
        return clusters_fit[-1][0], sorted_coordinates

    def _calc_rectangle_fit(self, v1, v2, v3, v4):

        angles = np.array([])
        for x, y, z in combinations([v1, v2, v3, v4], 3):
            # Calculate the lengths of the edges of this triangle.
            a = np.linalg.norm(y - x)
            b = np.linalg.norm(z - y)
            c = np.linalg.norm(x - z)

            # Use the cosine rule to get the angles.
            alpha = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            beta = np.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c))
            gamma = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

            angles = np.append(angles, np.array([alpha, beta, gamma]))

        #
        angles.sort()

        # 9 of angles should be +/- 45 and the 3 should be +/- 90 deg.
        # So we calculate the root mean square after subtracting 45 and 90 degrees.
        rms_45 = np.sqrt(np.mean(angles[:9] - np.deg2rad(45)) ** 2)
        rms_90 = np.sqrt(np.mean(angles[9:] - np.deg2rad(90)) ** 2)

        # A good score has low rms values.
        # return -rms_45 - rms_90

        # A simple test would be to check if the four shortest distances between the vertices are the same length.
        lengths_between_vertices = []

        # Loop through all combinations and get the distance between them.
        for a, b in combinations([v1, v2, v3, v4], 2):
            lengths_between_vertices.append(np.linalg.norm(b - a))

        # A good rectangle is when the four shortest edges are about the same length, and when the two longest
        # (the diagonals) are sqrt(2) longer.
        lengths_between_vertices.sort()

        std_four_shortest = np.std(lengths_between_vertices[:4]) / np.average(lengths_between_vertices[:4])
        std_two_longest = np.std(lengths_between_vertices[-2:]) / np.average(lengths_between_vertices[-2:])
        ratio_long_short = np.average(lengths_between_vertices[-2:]) / np.average(lengths_between_vertices[:4])

        return -std_four_shortest - std_two_longest - rms_45 - rms_90

    def _estimate_mask(self, score, coords, shape):

        # Start with an empty mask and x and y coordinates where we thing the gate is.
        mask = np.zeros(shape)
        gate_coords = np.empty((0, 2), dtype=int)

        # Keep track of all lengths to estimate the width of the gate.
        lengths = np.zeros(len(coords))

        # We'll estimate the mask by finding the edges of the rectangle and then moving them around by the width of the
        # gate. The width will be estimated using the length of the edges.
        # Loop through the coordinates clockwise.
        for i in range(len(coords)):

            # Also select the next coordinate.
            j = (i + 1) % 4

            # Calculate the length of this edge.
            lengths[i] = np.linalg.norm(coords[j] - coords[i])

            # Find the points that are on the line between these two coordinates, using a simple linear fit.
            slope = (coords[j, 1] - coords[i, 1]) / (coords[j, 0] - coords[i, 0])

            # If the slope is high, and we draw per x-coordinate, it skips a lot of points.
            if np.abs(slope) < 1:
                xs = np.arange(coords[i, 0], coords[j, 0], 1 if coords[i, 0] < coords[j, 0] else -1, dtype=int)
                ys = np.asarray(slope * (xs - coords[i, 0]) + coords[i, 1], dtype=int)
            else:
                slope = (coords[j, 0] - coords[i, 0]) / (coords[j, 1] - coords[i, 1])
                ys = np.arange(coords[i, 1], coords[j, 1], 1 if coords[i, 1] < coords[j, 1] else -1, dtype=int)
                xs = np.asarray(slope * (ys - coords[i, 1]) + coords[i, 0], dtype=int)

            # All points on this fit will now be masked with the score.
            mask[xs, ys] = score

        # Now we have a thin line that we want to make wider.
        halve_width = int(np.rint(np.mean(lengths) / 2 * self.gate_width_over_length_ratio))
        rectangle_coords = (mask == score)
        for offset in range(-halve_width, halve_width):
            if offset == 0:
                continue
            mask[np.roll(rectangle_coords, offset, axis=0)] = score
            mask[np.roll(rectangle_coords, offset, axis=1)] = score

        # Return the transpose because in the image the rows are on the y axis.
        return mask.transpose()
