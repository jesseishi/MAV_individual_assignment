# With this file a lot of images may be processed with varying parameters.

# Imports.
import cv2
import os
from src.final.GateDetector import GateDetector
from src.final.TestGateDetector import TestGateDetector
import pandas as pd
import numpy as np
import time

# Load the csv file with the true corner coordinates.
# The coordinates are stored from the top left and then the rest is clockwise.
data_folder = os.path.abspath(os.path.join(os.curdir, '../..', 'WashingtonOBRace'))
df_coords = pd.read_csv(os.path.join(data_folder, 'corners.csv'),
                        names=["im_name", "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"])

# Initialise the output dataframe for this entire batch.
results_batch = pd.DataFrame(columns=["eps", "min_samples", "n_images", "n_gates",
                                      "total_computation_time", "total_good_detection",
                                      "ave_computation_time", "ave_good_detection"])


# For more accurate timing of the gate detector we can increase this number so it runs it multiple times on the same
# image.
n_repeat_detections = 4

# Select which images we want to look at, there are 308 in total.
start_images = 0
stop_images = 154
unique_image_names = np.unique(df_coords["im_name"])[start_images:stop_images]

# Loop through different settings.
run_i = 0
t0 = time.time()
for eps in range(8, 19, 1):
    for min_samples in range(10, 23, 1):
        print('run #{}: eps: {}, min_samples: {}'.format(run_i, eps, min_samples))

        # Define parameters and set up the gate detector and gate detector tester.
        orb_params = {"edgeThreshold": 0}
        dbscan_params = {"eps": eps, "min_samples": min_samples}
        test_gate_detector_params = {"max_coordinate_error": 100}

        gate_detector = GateDetector(orb_params, dbscan_params)
        test_gate_detector = TestGateDetector(**test_gate_detector_params)

        # Initialize the output dataframe for these settings, that will store the results per image.
        results_im = pd.DataFrame(columns=["im_name", "n_image", "n_gates", "computation_time", "good_detection"])

        # Loop through all the images.
        # Some images have multiple gates, in which case it has multiple rows. So we loop through the unique image
        # names, and then loop through the possible gate locations.
        for im_name in unique_image_names:

            # Load the image.
            im = cv2.imread(os.path.join(data_folder, im_name), 0)

            # Sometimes the gate detector cannot find enough clusters and raises an error so we try to catch those
            # errors.
            try:

                # Keep track of time. If n_repeat_detections is bigger than 0 we'll repeat the detection step and
                # average out the time it takes.
                tic = time.time()
                for _ in range(n_repeat_detections + 1):
                    detect_coords = gate_detector.detect_gate(im)
                toc = time.time()
                computation_time = (toc - tic) / (n_repeat_detections + 1)

                # The algorithm can only detect one gate, so we should loop through the possibilities to see if we can
                # find a match.
                found_good_match = False
                n_gates_in_this_image = 0
                for _, row in df_coords[df_coords["im_name"] == im_name].iterrows():
                    n_gates_in_this_image += 1

                    # Get the coordinates of this gate from top left and then clockwise.
                    real_coords = np.array([[row["tl_x"], row["tl_y"]],
                                            [row["tr_x"], row["tr_y"]],
                                            [row["br_x"], row["br_y"]],
                                            [row["bl_x"], row["bl_y"]]])

                    # Check if it is a good match.
                    found_good_match = test_gate_detector.check_coordinates(detect_coords, real_coords)

                    # If we found a true positive, we don't need to check the other gates anymore.
                    if found_good_match:
                        break

                results_im = results_im.append({"im_name": im_name, "n_image": 1, "n_gates": n_gates_in_this_image,
                                                "computation_time": computation_time,
                                                "good_detection": found_good_match},
                                               ignore_index=True)

            except Exception as e:

                # Print out the error.
                print('{}, eps: {}, min_sample: {}'.format(im_name, eps, min_samples))
                print(e)

                # Add to the results anyway, then we can see on which scripts it didn't work.
                # Below, when we compute the average computation time, this won't be affected. While the average
                # detection rate will be affected.
                results_im = results_im.append({"im_name": im_name, "n_image": 1, "n_gates": np.nan,
                                                "computation_time": np.nan,
                                                "good_detection": False},
                                               ignore_index=True)
            finally:
                pass

        # Now that we've looped through all the images, we want to save the results of this run.
        results_im.to_csv(os.path.join('../', 'results', 'eps{}-min_samples{}.csv'.format(eps, min_samples)))

        # We also want to add the some metrics of this entire run to our batch results..
        results_batch = results_batch.append({"eps": eps,
                                              "min_samples": min_samples,
                                              "n_images": results_im["n_image"].sum(),
                                              "total_gates": results_im["n_gates"].sum(),
                                              "total_computation_time": results_im["computation_time"].sum(),
                                              "total_good_detection": results_im["good_detection"].sum(),
                                              "ave_computation_time": results_im["computation_time"].mean(),
                                              "ave_good_detection": results_im["good_detection"].mean()},
                                             ignore_index=True)
        run_i += 1

# Store the results of this batch.
results_batch.to_csv(os.path.join('../', 'results', 'batch.csv'))

# Report on the total time spent.
t1 = time.time()
print("Batch processed took {} seconds".format(t1-t0))
