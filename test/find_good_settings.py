# Make ROC curves.

# Imports.
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# Load data from the batch runner.
results_folder = os.path.abspath(os.path.join(os.curdir, '..', 'results'))
batch_df = pd.read_csv(os.path.join(results_folder, 'batch.csv'), index_col=0)

# Check the best combination of eps and min samples.
heat_ave_detection = batch_df.pivot('eps', 'min_samples', 'ave_good_detection')
heat_ave_computation = batch_df.pivot('eps', 'min_samples', 'ave_computation_time')

# Plot it.
plt.figure()
sns.heatmap(heat_ave_detection, annot=True, cmap="coolwarm", vmin=0, vmax=1)
plt.title('Average successful closest gate detection')

plt.figure()
sns.heatmap(heat_ave_computation, annot=True, cmap="coolwarm_r")
plt.title('Average computation time')