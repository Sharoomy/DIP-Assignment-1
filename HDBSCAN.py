#imports
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from astropy.io import fits
from astropy.table import Table
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




from sklearn.preprocessing import StandardScaler


#settings
min_cluster_size = 200
min_samples = 20


#load data
path = Path("/user/home/dl23921/ass1/DIP_2026")

#use catalog of galaxy morphology
morph_cat = "EUC_MER_FINAL-MORPH-CAT_TILE102159484-AE979F_20241026T003045.182961Z_00.00.fits"

#select key features
features = ["CONCENTRATION", "ASYMMETRY", "SMOOTHNESS", "GINI", "MOMENT_20"]

with fits.open(path / morph_cat) as hdul:
    morph_cat_table = Table(hdul[1].data)


#selects numerical collumns and drops constant collumns
morph_cat_df = pd.DataFrame(
    {f: np.asarray(morph_cat_table[f], dtype=float) for f in features}
)
morph_cat_df = morph_cat_df.replace([np.inf, -np.inf], np.nan).dropna()
print("Rows and columns after cleaning:", morph_cat_df.shape)

#select most important original features
X_selected = StandardScaler().fit_transform(morph_cat_df[features])
top_features = features

#train hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples)
labels = clusterer.fit_predict(X_selected)

#evaluate hdbscan
labels_s = pd.Series(labels, index=morph_cat_df.index, name="cluster")

#evaluate hdbscan 

groups = labels_s.value_counts().index[:6]            # largest groups (-1 = noise)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for j, (ax, feat) in enumerate(zip(axes, top_features)):
    col = X_selected[:, j]
    lo, hi = np.percentile(col, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)                    

    for cluster in groups:
        data = col[labels == cluster]
        name = "noise" if cluster == -1 else f"Cluster {cluster}"
        ax.hist(data, bins=bins, density=True, histtype="step", lw=1.5,
                label=f"{name} (N={len(data):,})")

    ax.set_xlabel(f"{feat} (z-score)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)

axes[-1].set_visible(False)                           
plt.tight_layout()
plt.savefig("hdbscan_cluster_density.png", dpi=200)
plt.close(fig)