# Imports.
import cv2
import os
from src.final.GateDetector import GateDetector
from matplotlib import pyplot as plt


# Define parameters.
orb_params = {}
dbscan_params = {"eps": 10, "min_samples": 10}

# Load an image.
data_folder = os.path.abspath(os.path.join(os.curdir, '../..', 'WashingtonOBRace'))
im = cv2.imread(os.path.join(data_folder, 'img_132.png'), 0)

# Detect a gate.
gate_detector = GateDetector(orb_params, dbscan_params)
coords = gate_detector.detect_gate(im)

# Plot it.
plt.figure()
plt.imshow(im, 'gray')
plt.plot(coords[:, 0], coords[:, 1], 'kX', markersize=15)
plt.show()
