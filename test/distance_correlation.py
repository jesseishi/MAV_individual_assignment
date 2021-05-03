# Check how successful detections are correlated with gate distance for different settings.

# Imports.
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import numpy as np

# Load data from the batch runner.
results_folder = os.path.abspath(os.path.join(os.curdir, '..', 'results'))

# We want to make a heatmap with gate size bins on the y axis, eps on the x axis and the heat is the detection rate.
df = pd.DataFrame(columns=["setting", "gate_size_bin", "detection_rate"])
gate_size_bins = np.arange(0, 1001, 100)
for eps in range(8, 29):
    min_samples = 18

    df_setting = pd.read_csv(os.path.join(results_folder, 'eps{}-min_samples{}.csv'.format(eps, min_samples)),
                             index_col=0)

    # For each closest_gate_size store in which bin it is.
    df_setting["distance_bin"] = np.digitize(df_setting["closest_gate_size"], bins=gate_size_bins)
    # df["gate_size_bin"] = gate_size_bins[df_setting["distance_bin"]]

    # Find the average detection rate per bin.
    bin_rates = df_setting.groupby("distance_bin").mean()["good_detection"]
    bin_count = df_setting.groupby("distance_bin").count()["good_detection"]

    # Add these rows to the dataframe.
    df_rows = pd.DataFrame.from_dict({"eps": [eps for _ in range(len(bin_rates))],
                                      "gate_size_bin": bin_rates.index,
                                      "detection_rate": bin_rates.values * 100,  # x100 to get percentage
                                      "count": bin_count.values})
    df = df.append(df_rows, ignore_index=True, sort=False)


# Now pivot this dataframe to make a heatmap.
df_detection_rate = df.pivot("gate_size_bin", "eps", "detection_rate")
df_count = df.pivot("gate_size_bin", "eps", "count")

# Make the plot.
sns.heatmap(df_detection_rate)