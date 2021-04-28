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
batch_df["tpr"] = batch_df["tp"] / (batch_df["tp"] + batch_df["fn"])
batch_df["fpr"] = batch_df["fp"] / (batch_df["fp"] + batch_df["tn"])

# Check the best combination of eps and min samples.
heat_df = batch_df.pivot('eps', 'min_samples', 'tpr')
sns.heatmap(heat_df, annot=True, cmap="coolwarm", vmin=0, vmax=1)
# Nice cmpashttps://matplotlib.org/stable/tutorials/colors/colormaps.html: viridis, coolwarm,

# Fix eps and min_samples separately and make the ROC curve.
eps_fix = 8
min_samples_fix = 10

eps_fixed_df = batch_df[batch_df['eps'] == eps_fix]
plt.figure()
plt.plot(eps_fixed_df['fpr'], eps_fixed_df['tpr'], 'x')
plt.xlabel('1 - specificity')
plt.ylabel('Sensitivity')
plt.show()
