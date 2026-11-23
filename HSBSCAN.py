#imports
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

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

morph_cat_df = morph_cat_df.replace([np.inf, -np.inf], np.nan)
morph_cat_df = morph_cat_df.dropna()

val_df = pd.DataFrame(
    {f: np.asarray(morph_cat_table[f])[morph_cat_df.index] for f in val_features},
    index=morph_cat_df.index,
)


#train hdbscan
scaler = StandardScaler()
test_data = scaler.fit_transform(morph_cat_df)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5
)

clusterer.fit(test_data)

labels = clusterer.labels_


#evaluate hdbscan

labels = clusterer.labels_
silhouette_avg = silhouette_score(test_data, labels)
ari = adjusted_rand_score(val_df['MAJOR_MERGER'], labels)

#evaluation metrics
print("Silhouette Coefficient: {:.2f}".format(silhouette_avg))
print("Adjusted Rand Index: {:.2f}".format(ari))

#visualise clusters with density plot

plt.figure(figsize=(10, 6))

for cluster in set(clusterer.labels_):
    cluster_data = test_data[clusterer.labels_ == cluster, 0]

    plt.hist(cluster_data, bins=40, density=True, alpha=0.5,
             label=f"Cluster {cluster}")

plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Density")
plt.title("HDBSCAN Cluster Density Distribution")
plt.legend()
plt.show()