#imports
import hdbscan

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from astropy.io import fits
from astropy.table import Table
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#settings
min_cluster_size = 200
min_samples = 20
n_imp_features = 10

#load data
path = Path("/user/home/dl23921/ass1/DIP_2026")

#use catalog of galaxy morphology
morph_cat = "EUC_MER_FINAL-MORPH-CAT_TILE102159484-AE979F_20241026T003045.182961Z_00.00.fits"


with fits.open(path / morph_cat) as hdul:
    morph_cat_table = Table(hdul[1].data)

#select key features
metadata_features = [col for col in morph_cat_table.colnames
    if (
        col.endswith("_ERR")
        or "FLAG" in col
        or "ITERATIONS" in col
        or "DURATION" in col
        or "REDUCED_CHI2" in col
        or col in [
            "OBJECT_ID",
            "RIGHT_ASCENSION",
            "DECLINATION"
        ]
    )
]

physical_features = [
    col for col in morph_cat_table.colnames
    if col not in metadata_features
]

#selects numerical collumns and drops constant collumns
morph_cat_df = pd.DataFrame( {f: np.asarray(morph_cat_table[f], dtype=float) for f in physical_features})

morph_cat_df = morph_cat_df.apply(pd.to_numeric, errors='coerce')
morph_cat_df = morph_cat_df.replace([np.inf, -np.inf], np.nan)

morph_cat_df = morph_cat_df.loc[:, morph_cat_df.nunique(dropna=True) > 1]

print("Rows and columns after cleaning:", morph_cat_df.shape)

# replace missing values with column medians
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(morph_cat_df)

morph_cat_df_imputed = pd.DataFrame(X_imputed, columns=morph_cat_df.columns,index=morph_cat_df.index)
#exclude insdentifying features

#scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

#run pca
pca_model = PCA(n_components=0.95)
X_pca = pca_model.fit_transform(X_scaled)

print("Number of PCA components:", pca_model.n_components_)
print(f"Explained variance for PCA 1 & 2:", pca_model.explained_variance_ratio_)
print(f"Explained variance total:", np.sum(pca_model.explained_variance_ratio_))

#calacute importance of features from pca loadings
feature_importance = np.sum(
    pca_model.components_**2 *
    pca_model.explained_variance_ratio_[:, np.newaxis],
    axis=0
)

#make dataframe of important features
feature_importance_df = pd.DataFrame({
    'Feature': morph_cat_df.columns,
    'Importance': feature_importance
})

#sort important features
feature_importance_df = feature_importance_df.sort_values(
    by='Importance',
    ascending=False
)

assert len(feature_importance) == len(morph_cat_df.columns)
assert len(feature_importance_df) == len(morph_cat_df.columns)

print("Shape of important features:", len(feature_importance))
print("Number of cleaned features:", len(morph_cat_df.columns))

#plot feature importance for top 20 features
plt.figure(figsize=(10, 8))

plt.barh(
    feature_importance_df['Feature'].head(20)[::-1],
    feature_importance_df['Importance'].head(20)[::-1]
)

plt.xlabel('PCA Feature Contribution')
plt.ylabel('Feature')
plt.title(f'Top {n_imp_features} Features from PCA')

plt.tight_layout()
plt.savefig('pca_imp_feat.png', dpi=200)

# select 20 most important original features
top_features = feature_importance_df['Feature'].head(
    n_imp_features
).tolist()

# extract selected features from scaled dataset
selected_indices = [
    list(morph_cat_df.columns).index(col)
    for col in top_features
]

X_selected = X_scaled[:, selected_indices]

#print checks after cleaning
print("Original catalogue rows:", len(morph_cat_table))
print("Rows after cleaning:", len(morph_cat_df))
print("Remaining missing values:", morph_cat_df.isna().sum().sum())

#train hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples)
labels = clusterer.fit_predict(X_selected)
scores = clusterer.outlier_scores_

#evaluate hdbscan
labels_s = pd.Series(labels, index=morph_cat_df.index, name="cluster")
is_noise = labels == -1
n_clusters = len(set(labels[~is_noise]))

print(f"\nClusters: {n_clusters}")
print(f"Noise points: {is_noise.sum():,} ({is_noise.mean():.1%})")
print("Largest groups (-1 = noise):")

#silhoutte score 

if n_clusters >= 2:
    keep = ~is_noise
    sil = silhouette_score(X_selected[keep], labels[keep],sample_size=min(1000, int(keep.sum())),random_state=22)
    print(f"Silhouette (non-noise, subsampled): {sil:.3f}")
else:
    print("Silhouette undefined: fewer than 2 clusters found")

#desnity plot for each feature

n_cols = 3 
n_rows = int(np.ceil(n_imp_features / n_cols))

groups = labels_s.value_counts().index[:6]            

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
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

for i in range(len(top_features), len(axes)):
    axes[i].set_visible(False)
                        
plt.tight_layout()
plt.savefig("hdbscan_cluster_density.png", dpi=200)
plt.close(fig)

#
threshold = pd.Series(scores).quantile(0.9)
morph_cat_df_imputed["anomaly"] = np.where(scores > threshold, -1, 1)

print(morph_cat_df_imputed["anomaly"].value_counts())

#make normal and anomaly dataframes
normal = morph_cat_df[morph_cat_df_imputed["anomaly"] == 1]
anomalies = morph_cat_df[morph_cat_df_imputed["anomaly"] == -1]

#visualise anomaly scores using density plot

#density plots of normal vs anomalous objects for every feature

fig, axes = plt.subplots(2, (n_imp_features//2), figsize=(16, 9))
axes = axes.flatten()

for ax, col in zip(axes, top_features):
    # 1. Get data for both groups
    data_norm = normal[col].dropna()
    data_anom = anomalies[col].dropna()
    
    # 2. Determine a robust range that includes both groups
    # We take the min/max of the actual data present in both
    all_data = pd.concat([data_norm, data_anom])
    if len(all_data) > 0:
        lo = all_data.min()
        hi = all_data.max()

        print(f"Feature: {col}")
        print(f"  Normal range: {normal[col].min():.2f} to {normal[col].max():.2f}")
        print(f"  Anomaly range: {anomalies[col].min():.2f} to {anomalies[col].max():.2f}")
        print(f"  Plotting bins: {lo:.2f} to {hi:.2f}")

        # Use more bins if the range is large, fewer if small
        bins = np.linspace(lo, hi, 80)

        ax.hist(data_norm, bins=bins, density=True, alpha=0.5,
                color="tab:green", label=f"Normal (N={len(data_norm):,})")
        ax.hist(data_anom, bins=bins, density=True, alpha=0.5,
                color="tab:red", label=f"Anomaly (N={len(data_anom):,})")
        
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    else:
        ax.set_title(f"{col} (No Data)")

plt.tight_layout()
plt.savefig("anomaly_density_hdbscan.png", dpi=200)

