# Imports.
import numpy as np


# This class takes the output from the GateDetector and compares it to the true coordinates of the corners.
# So we can check if it detected it correctly.
class TestGateDetector:
    def __init__(self, max_total_error):
        self.max_total_error = max_total_error

    # TODO: how is a TP defined.
    def is_true_positive(self, detect_coords, real_coords, return_error=False):

        # total_error = 0

        # Assume that all coordinates are sorted from top left and then clockwise.
        # TODO: Make it such that the order doesn't matter. By sorting the x coordinates and then the y coordinates and
        #  then taking the norm.
        real_xs = real_coords[:, 0].copy()
        real_ys = real_coords[:, 1].copy()
        detect_xs = detect_coords[:, 0].copy()
        detect_yx = detect_coords[:, 1].copy()

        real_xs.sort()
        real_ys.sort()
        detect_xs.sort()
        detect_yx.sort()

        total_error = np.linalg.norm(real_xs - detect_xs) + np.linalg.norm(real_ys - detect_yx)
        # for detect_xy, real_xy in zip(detect_coords, real_coords):
        #     total_error += np.linalg.norm(real_xy - detect_xy)

        if return_error:
            return total_error < self.max_total_error, total_error
        else:
            return total_error < self.max_total_error
