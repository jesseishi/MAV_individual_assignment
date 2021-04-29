# Imports.
import numpy as np


# This class takes a GateDetector and compares its output to the true coordinates of the corners and mask.
# So we can check if it detected it correctly.
class TestGateDetector:
    def __init__(self, max_coordinate_error):
        self.max_coordinate_error = max_coordinate_error

    def check_coordinates(self, detect_coords, real_coords, return_error=False):

        # All coordinates are sorted starting in the top left and then going clockwise.
        # So we can simply loop through them and add all the errors.
        total_error = 0
        for a, b in zip(detect_coords, real_coords):
            total_error += np.linalg.norm(b - a)

        # If the total error is below a certain threshold we count is as a good detection.
        # Note that this is different from a true positive as this is not a binary classification task.
        is_good_detection = total_error < self.max_coordinate_error

        if return_error:
            return is_good_detection, total_error
        else:
            return is_good_detection
