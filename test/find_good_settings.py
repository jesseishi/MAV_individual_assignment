# Find good setting of the algorithm.

# Imports.
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import numpy as np

# Load data from the batch runner.
results_folder = os.path.abspath(os.path.join(os.curdir, '..', 'results'))
batch_df = pd.read_csv(os.path.join(results_folder, 'batch training data.csv'), index_col=0)
batch_df["ave_computation_speed"] = 1/batch_df["ave_computation_time"]
batch_df["ave_detection_percentage"] = batch_df["ave_good_detection"] * 100
batch_df["ave_good_detection_speed"] = batch_df["ave_computation_speed"] * batch_df["ave_good_detection"]

# Check the best combination of eps and min samples.
heat_ave_detection = batch_df.pivot('eps', 'min_samples', 'ave_detection_percentage')
heat_ave_computation = batch_df.pivot('eps', 'min_samples', 'ave_computation_speed')
heat_ave_detection_ps = batch_df.pivot('eps', 'min_samples', 'ave_good_detection_speed')

# Plot it.
# Annotating the heatmap is nice, but it's too much to have it everywhere, so make a [0, 1, 0, 1, ... array].
mask = np.zeros(heat_ave_detection.shape).flatten()
mask[1::2] = 1
mask = mask.reshape(heat_ave_detection.shape)
plt.figure()
sns.heatmap(heat_ave_detection, annot=False, cmap="coolwarm", vmin=0, vmax=100,
            linewidths=0.0, xticklabels=2,
            yticklabels=2, fmt='.0f', cbar=False)
sns.heatmap(heat_ave_detection, annot=True, cmap="coolwarm", vmin=0, vmax=100,
            linewidths=0.0, xticklabels=2,
            yticklabels=2, fmt='.0f', mask=mask)
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
plt.title('Average gate detection rate [%]')

plt.figure()
sns.heatmap(heat_ave_computation, annot=False, cmap="coolwarm", vmin=0, linewidths=0.0, xticklabels=2,
            yticklabels=2, fmt='.0f', cbar=False)
sns.heatmap(heat_ave_computation, annot=True, cmap="coolwarm", vmin=0, linewidths=0.0, xticklabels=2,
            yticklabels=2, fmt='.0f', mask=mask)
plt.title('Average computation speed [images / s]')


# Check the errors of the score we like.
eps = 8
min_samples = 24
df_settings = pd.read_csv(os.path.join(results_folder, 'eps{}-min_samples{}.csv'.format(eps, min_samples)), index_col=0)

plt.figure()
df_settings["error"].hist(bins=np.arange(0, np.max(df_settings["error"]), 5))
