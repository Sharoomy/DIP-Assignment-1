#isolation forest

#imports
from sklearn.ensemble import IsolationForest

from astropy.io import fits
from astropy.table import Table
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load data

path = Path("/user/home/dl23921/ass1/DIP_2026")

morph_cat = "EUC_MER_FINAL-MORPH-CAT_TILE102159484-AE979F_20241026T003045.182961Z_00.00.fits"

with fits.open(path / morph_cat) as hdul:
    morph_cat_table = Table(hdul[1].data)

#select wanted features

numerical_features = [
    "CONCENTRATION",
    "ASYMMETRY",
    "SMOOTHNESS",
    "GINI",
    "MOMENT_20"
]

val_features = [
    "MAJOR_MERGER",
    "PECULIAR_YES",
    "CLUMP_COUNT_UNUSUAL_ANY_THRESHOLD"
]

#make all values numerical and drop na
morph_cat_df = pd.DataFrame(
    {f: np.asarray(morph_cat_table[f], dtype=float) for f in numerical_features}
)

val_df = pd.DataFrame(
    {f: np.asarray(morph_cat_table[f])[morph_cat_df.index] for f in val_features},
    index=morph_cat_df.index,
)

morph_cat_df = morph_cat_df.replace([np.inf, -np.inf], np.nan)
morph_cat_df = morph_cat_df.dropna()


print(morph_cat_df.shape)
print(morph_cat_df.head())

#parameters for isolation forest
contamination = 0.05  
n_estimators = 100
sample_size = 256

morph_mat = morph_cat_df[numerical_features]

#train isolation forest
iso_forest = IsolationForest(contamination=contamination, 
                             n_estimators=n_estimators, 
                             max_samples=sample_size, 
                             random_state=42)

iso_forest.fit(morph_mat)

#calculate anomaly scores and predictions

morph_cat_df["anomaly_score"] = iso_forest.decision_function(morph_mat)
morph_cat_df["anomaly"] = iso_forest.predict(morph_mat)

print(morph_cat_df["anomaly"].value_counts())

#make normal and anomaly dataframes
normal = morph_cat_df[morph_cat_df["anomaly"] == 1]
anomalies = morph_cat_df[morph_cat_df["anomaly"] == -1]

#visualise anomaly scores using density plot

#density plots of normal vs anomalous objects for every feature
plot_cols = numerical_features + ["anomaly_score"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax, col in zip(axes, plot_cols):
   
    lo, hi = np.percentile(morph_cat_df[col], [0.1, 99.9])
    bins = np.linspace(lo, hi, 80)

    ax.hist(normal[col], bins=bins, density=True, alpha=0.5,
            color="tab:green", label=f"Normal (N={len(normal):,})")
    ax.hist(anomalies[col], bins=bins, density=True, alpha=0.5,
            color="tab:red", label=f"Anomaly (N={len(anomalies):,})")

    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend()

#decision threshold line at 0
axes[-1].axvline(0, color="black", ls="--", lw=1)

plt.tight_layout()
plt.savefig("anomaly_density.png", dpi=200)

#valide using the val features