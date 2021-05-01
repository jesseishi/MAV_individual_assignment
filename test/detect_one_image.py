# Imports.
import cv2
import os
from src.final.GateDetector import GateDetector
from src.final.TestGateDetector import TestGateDetector
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np


# Define parameters.
orb_params = {"edgeThreshold": 0}
dbscan_params = {"eps": 13, "min_samples": 18}
test_gate_detector_params = {"max_coordinate_error": 75}

# Load an image and its coordinates.
im_number = 279
im_name = 'img_{}.png'.format(im_number)
mask_name = 'mask_{}.png'.format(im_number)
data_folder = os.path.abspath(os.path.join(os.curdir, '../..', 'WashingtonOBRace'))
im = cv2.imread(os.path.join(data_folder, im_name), 0)
mask = cv2.imread(os.path.join(data_folder, mask_name), 0)
df_coords = pd.read_csv(os.path.join(data_folder, 'corners.csv'),
                        names=["im_name", "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"])

# Detect a gate.
gate_detector = GateDetector(orb_params, dbscan_params)
detect_coords, mask_hat = gate_detector.detect_gate(im, return_mask=True)

# Start the plot.
plt.figure()
plt.imshow(im, 'gray')

# Plot the clusters.
for cluster in np.unique(gate_detector.y_hat):

    # get row indexes for samples with this cluster
    row_ix = np.where(gate_detector.y_hat == cluster)

    # Create scatter of these samples
    plt.scatter(gate_detector.X[row_ix, 0], gate_detector.X[row_ix, 1])


# Check if it is good and plot the real coordinates.
test_gate_detector = TestGateDetector(**test_gate_detector_params)
found_true_positive = False
for i, row in df_coords[df_coords["im_name"] == im_name].iterrows():

    # Get the coordinates of this gate.
    real_coords = np.array([[row["tl_x"], row["tl_y"]],
                            [row["tr_x"], row["tr_y"]],
                            [row["br_x"], row["br_y"]],
                            [row["bl_x"], row["bl_y"]]])

    # Check if it is a true positive.
    is_true_positive, error = test_gate_detector.check_coordinates(detect_coords, real_coords, return_error=True)
    if is_true_positive:
        found_true_positive = True

    plt.plot(real_coords[:, 0], real_coords[:, 1], 'X', markersize=10, label="total error: {}".format(error))

plt.plot(detect_coords[:, 0], detect_coords[:, 1], 'X', color='green' if found_true_positive else 'red', markersize=20)
plt.legend()
plt.show()

plt.figure()
plt.imshow(mask_hat, 'Blues', alpha=0.5)
plt.imshow(mask, 'Reds', alpha=0.5)
plt.show()

# Plot the whole process.
fig, axs = plt.subplots(1, 3, figsize=(7, 2))
axs[0].imshow(im, 'gray')
axs[1].imshow(im, 'gray')
axs[2].imshow(im, 'gray')

# Plot the clusters.
for cluster in np.unique(gate_detector.y_hat):

    # get row indexes for samples with this cluster
    row_ix = np.where(gate_detector.y_hat == cluster)

    # Create scatter of these samples
    axs[0].plot(gate_detector.X[row_ix, 0], gate_detector.X[row_ix, 1], 'bx')
    axs[1].scatter(gate_detector.X[row_ix, 0], gate_detector.X[row_ix, 1], marker='x')

axs[2].plot(detect_coords[:, 0], detect_coords[:, 1], 'X', color='green', markersize=15)

for i in range(3):
    axs[i].set_title('step {}'.format(i+1))
    axs[i].set_ylim([np.shape(im)[0], 0])
    axs[i].set_xlim([0, np.shape(im)[1]])
    axs[i].xaxis.set_major_locator(ticker.NullLocator())
    axs[i].yaxis.set_major_locator(ticker.NullLocator())
