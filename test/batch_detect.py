# With this file a lot of images may be processed with varying parameters.
# From the results ROC curves may be constructed.


# Imports.
import cv2
import os
from src.final.GateDetector import GateDetector
from src.final.TestGateDetector import TestGateDetector
import pandas as pd
import numpy as np


# Load the csv file with the true corner coordinates.
# The coordinates are stored like: top_left_x, top_left_y, and then the rest is clockwise.
data_folder = os.path.abspath(os.path.join(os.curdir, '../..', 'WashingtonOBRace'))
df_coords = pd.read_csv(os.path.join(data_folder, 'corners.csv'), names=["im_name",
                                                                         "tl_x", "tl_y",
                                                                         "tr_x", "tr_y",
                                                                         "br_x", "br_y",
                                                                         "bl_x", "bl_y"])

# Define parameters and set up the gate detector and gate detector tester.
orb_params = {}
dbscan_params = {"eps": 10, "min_samples": 10}
test_gate_detector_params = {"max_total_error": 50}
max_images = 100

gate_detector = GateDetector(orb_params, dbscan_params)
test_gate_detector = TestGateDetector(**test_gate_detector_params)

# Loop through all the images.
# Some images have multiple gates, in which case it has multiple rows. So we loop through the unique image names, and
# then loop through the possible gate locations.

# Make a dictionary to store the results like {"im_name": found_true_positive}
result_dict = {}
for i, im_name in enumerate(np.unique(df_coords["im_name"])):

    # Load the image.
    im = cv2.imread(os.path.join(data_folder, im_name), 0)

    # Try to detect a gate.
    detect_coords = gate_detector.detect_gate(im)

    # Loop through the possible gates and see if we detected one correctly.
    # TODO: this assumes that the algorithm can only detect one gate.
    found_true_positive = False
    for i, row in df_coords[df_coords["im_name"] == im_name].iterrows():

        # Get the coordinates of this gate.
        real_coords = np.array([[row["tl_x"], row["tl_y"]],
                                [row["tr_x"], row["tr_y"]],
                                [row["br_x"], row["br_y"]],
                                [row["bl_x"], row["bl_y"]]])

        # Check if it is a true positive.
        found_true_positive = test_gate_detector.is_true_positive(detect_coords, real_coords)

        # If we found a true positive, we don't need to check the other gates anymore.
        if found_true_positive:
            break

    result_dict[im_name] = found_true_positive

    if i > max_images:
        break

# Save the results.
result_df = pd.DataFrame.from_dict(result_dict, orient='index', columns=["found_true_positive"])
result_df.to_csv(os.path.join('../', 'results', 'joe.csv'))
