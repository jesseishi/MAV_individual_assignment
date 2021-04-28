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
df_coords = pd.read_csv(os.path.join(data_folder, 'corners.csv'),
                        names=["im_name", "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"])

results_batch = pd.DataFrame(columns=["eps", "min_samples", "n_images", "total_gates", "tp", "fp", "tn", "fn"])
run_i = 0
max_images = 100
for eps in range(8, 21, 2):
    for min_samples in range(10, 21, 2):
        print('run #{}: eps: {}, min_samples: {}'.format(run_i, eps, min_samples))

        # Encapsulate everything in a big try-except block.
        # Define parameters and set up the gate detector and gate detector tester.
        orb_params = {"edgeThreshold": 0}
        dbscan_params = {"eps": eps, "min_samples": min_samples}
        test_gate_detector_params = {"max_total_error": 50}

        gate_detector = GateDetector(orb_params, dbscan_params)
        test_gate_detector = TestGateDetector(**test_gate_detector_params)

        # Reset the total amount of true positives and false positives for this run.
        # TODO: distinguish between total tp, fp, tn, and fn for the batch and one image.
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        total_gates = 0

        # Loop through all the images.
        # Some images have multiple gates, in which case it has multiple rows. So we loop through the unique image
        # names, and then loop through the possible gate locations.

        # Make a dictionary to store the results like {"im_name": found_true_positive}
        result_set_params_dict = {}
        for i, im_name in enumerate(np.unique(df_coords["im_name"])):

            # Encapsulate everything in a try-except block.
            try:
                # Load the image.
                im = cv2.imread(os.path.join(data_folder, im_name), 0)

                # Try to detect a gate.
                detect_coords = gate_detector.detect_gate(im)

                # Loop through the possible gates and see if we detected one correctly.
                # TODO: this assumes that the algorithm can only detect one gate.
                found_true_positive = False
                n_gates_in_this_image = 0  # This will be iterated on.
                for _, row in df_coords[df_coords["im_name"] == im_name].iterrows():
                    n_gates_in_this_image += 1
                    total_gates += 1

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

                # If we found a true positive increase the counter.
                if found_true_positive:
                    tp += 1
                    fn += n_gates_in_this_image - 1
                else:
                    fp += 1
                    fn += n_gates_in_this_image

                # We assume there are always 3 potential gates in the image, since we can only detect 1 at most, the
                # rest will be true negatives.
                tn += 3 - n_gates_in_this_image

                result_set_params_dict[im_name] = found_true_positive

            except Exception as e:
                print(e)

                # We count a failure as a false positive.
                fp += 1
            finally:

                if i == max_images - 1:
                    break

        # Save the results of this run.
        results_run = pd.DataFrame.from_dict(result_set_params_dict, orient='index', columns=["found_true_positive"])
        results_run.to_csv(os.path.join('../', 'results', 'eps{}-min_samples{}.csv'.format(eps, min_samples)))

        # Store the results of this run.
        results_batch = results_batch.append({"eps": eps, "min_samples": min_samples, "n_images": i+1,
                                              "total_gates": total_gates,
                                              "tp": tp, "fp": fp, "tn": tn, "fn": fn},
                                             ignore_index=True)
        run_i += 1

# Store the results of this batch.
results_batch.to_csv(os.path.join('../', 'results', 'batch.csv'))
