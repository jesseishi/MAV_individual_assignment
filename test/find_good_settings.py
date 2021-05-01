# Make ROC curves.

# Imports.
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import numpy as np

# Load data from the batch runner.
results_folder = os.path.abspath(os.path.join(os.curdir, '..', 'results'))
batch_df = pd.read_csv(os.path.join(results_folder, 'batch.csv'), index_col=0)
batch_df["ave_computation_speed"] = 1/batch_df["ave_computation_time"]
batch_df["ave_detection_percentage"] = batch_df["ave_good_detection"] * 100

# Check the best combination of eps and min samples.
heat_ave_detection = batch_df.pivot('eps', 'min_samples', 'ave_detection_percentage')
heat_ave_computation = batch_df.pivot('eps', 'min_samples', 'ave_computation_speed')

# Plot it.
# Annotating the heatmap is nice, but it's too much to have it everywhere, so make a [0, 1, 0, 1, ... array].
mask = np.zeros(heat_ave_detection.shape).flatten()
mask[1::2] = 1
mask = mask.reshape(heat_ave_detection.shape)
plt.figure()
sns.heatmap(heat_ave_detection, annot=True, cmap="coolwarm", vmin=0, vmax=100,
            linewidths=0.5, xticklabels=2,
            yticklabels=2, fmt='.0f')
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
plt.title('Average gate detection rate [%]')

plt.figure()
sns.heatmap(heat_ave_computation, annot=True, cmap="coolwarm", vmin=0, linewidths=0.5, xticklabels=2,
            yticklabels=2, fmt='.0f')
plt.title('Average computation speed [images / s]')


# Check the errors of the score we like.
eps = 13
min_samples = 18
df_settings = pd.read_csv(os.path.join(results_folder, 'eps{}-min_samples{}.csv'.format(eps, min_samples)), index_col=0)

plt.figure()
df_settings["error"].hist(bins=np.arange(0, np.max(df_settings["error"]), 5))

# Also check if the detection rate is correlated with size of the gate for these settings.
plt.figure(figsize=(5, 2))
plt.plot(df_settings["closest_gate_size"], df_settings["good_detection"], 'x')
plt.yticks([0, 1])
plt.ylim([-0.5, 1.5])