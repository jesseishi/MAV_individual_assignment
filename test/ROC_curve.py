# Make ROC curves.

# Imports.
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# Load data.
results_folder = os.path.abspath(os.path.join(os.curdir, '..', 'results'))
batch_df = pd.read_csv(os.path.join(results_folder, 'batch.csv'), index_col=0)

# Make fractions of true/false positive amounts.
batch_df["tp_f"] = batch_df["tp"] / batch_df["n_images"]
batch_df["fp_f"] = batch_df["fp"] / batch_df["n_images"]

# Check the best combination of eps and min samples.
heat_df = batch_df.pivot('eps', 'min_samples', 'tp_f')
sns.heatmap(heat_df, annot=True, cmap="YlGnBu")
