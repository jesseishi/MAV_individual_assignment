# Imports.
import numpy as np


# This class takes the output from the GateDetector and compares it to the true coordinates of the corners.
# So we can check if it detected it correctly.
class TestGateDetector:
    def __init__(self, max_total_error):
        self.max_total_error = max_total_error

    # TODO: how is a TP defined.
    def is_true_positive(self, detect_coords, real_coords):

        total_error = 0

        # Assume that all coordinates are sorted from left to right, top to bottom.
        for detect_xy, real_xy in zip(detect_coords, real_coords):
            total_error += np.linalg.norm(real_xy - detect_xy)

        return total_error < self.max_total_error
