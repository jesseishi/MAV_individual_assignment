# Imports.
import cv2
import os
from src.final.GateDetector import GateDetector
from src.final.TestGateDetector import TestGateDetector
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


# Define parameters.
orb_params = {}
dbscan_params = {"eps": 10, "min_samples": 10}
test_gate_detector_params = {"max_total_error": 50}

# Load an image and its coordinates.
im_name = 'img_139.png'
data_folder = os.path.abspath(os.path.join(os.curdir, '../..', 'WashingtonOBRace'))
im = cv2.imread(os.path.join(data_folder, im_name), 0)
df_coords = pd.read_csv(os.path.join(data_folder, 'corners.csv'), names=["im_name",
                                                                         "tl_x", "tl_y",
                                                                         "tr_x", "tr_y",
                                                                         "br_x", "br_y",
                                                                         "bl_x", "bl_y"])

# Detect a gate.
gate_detector = GateDetector(orb_params, dbscan_params)
detect_coords = gate_detector.detect_gate(im)

# Check if it is good and plot the real coordinates.
plt.figure()

test_gate_detector = TestGateDetector(**test_gate_detector_params)
found_true_positive = False
for i, row in df_coords[df_coords["im_name"] == im_name].iterrows():

    # Get the coordinates of this gate.
    real_coords = np.array([[row["tl_x"], row["tl_y"]],
                            [row["tr_x"], row["tr_y"]],
                            [row["br_x"], row["br_y"]],
                            [row["bl_x"], row["bl_y"]]])

    # Check if it is a true positive.
    is_true_positive, error = test_gate_detector.is_true_positive(detect_coords, real_coords, return_error=True)
    if is_true_positive:
        found_true_positive = True

    plt.plot(real_coords[:, 0], real_coords[:, 1], 'X', markersize=10, label="total error: {}".format(error))

plt.imshow(im, 'gray')
plt.plot(detect_coords[:, 0], detect_coords[:, 1], 'X', color='green' if found_true_positive else 'red', markersize=10)
plt.legend()
plt.show()