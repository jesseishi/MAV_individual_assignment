# This file is used to generate the data used to make a single ROC curves.

# Imports.
import cv2
import os
from src.final.GateDetector import GateDetector
from src.final.TestGateDetector import TestGateDetector
import pandas as pd
import numpy as np
import time

# Load the csv file with the true corner coordinates since it has the image names.
data_folder = os.path.abspath(os.path.join(os.curdir, '../..', 'WashingtonOBRace'))
df_coords = pd.read_csv(os.path.join(data_folder, 'corners.csv'),
                        names=["im_name", "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"])

# Select which images we want to look at, there are 308 in total.
start_images = 0
stop_images = 50
unique_image_names = np.unique(df_coords["im_name"])[start_images:stop_images]

# Define parameters and set up the gate detector and gate detector tester.
orb_params = {"edgeThreshold": 0}
dbscan_params = {"eps": 21, "min_samples": 25}
test_gate_detector_params = {"max_coordinate_error": 75}

gate_detector = GateDetector(orb_params, dbscan_params)
test_gate_detector = TestGateDetector(**test_gate_detector_params)

all_masks = np.array([])
all_masks_hat = np.array([])

for im_name in unique_image_names:

    # Load the image and its mask.
    im = cv2.imread(os.path.join(data_folder, im_name), 0)
    mask = cv2.imread(os.path.join(data_folder, im_name.replace('img', 'mask')), 0)

    try:
        _, mask_hat = gate_detector.detect_gate(im, return_mask=True)

    # If the code above fails, we return an empty mask, because we detected nothing.
    except Exception as e:
        print(e)
        mask_hat = np.zeros_like(mask)

    all_masks = np.append(all_masks, mask.flatten())
    all_masks_hat = np.append(all_masks_hat, mask_hat.flatten())

np.save(os.path.join('../', 'results', 'all_masks.npy'), all_masks)
np.save(os.path.join('../', 'results', 'all_masks_hat.npy'), all_masks_hat)
