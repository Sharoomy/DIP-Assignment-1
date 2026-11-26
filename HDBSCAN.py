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


import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#settings
ran_state = 64
min_cluster_size = 10
min_samples = 5
n_imp_features = 10

#load data
path = Path("/user/home/dl23921/ass1/DIP_2026")

#use catalog of galaxy morphology
morph_cat = "EUC_MER_FINAL-MORPH-CAT_TILE102159484-AE979F_20241026T003045.182961Z_00.00.fits"

with fits.open(path / morph_cat) as hdul:
    morph_cat_table = Table(hdul[1].data)


#selects numerical collumns and drops constant collumns
morph_cat_df = morph_cat_table.to_pandas()

morph_cat_df = morph_cat_df.apply(pd.to_numeric, errors='coerce')
morph_cat_df = morph_cat_df.replace([np.inf, -np.inf], np.nan)

#drops collumns with nans above 25%
morph_cat_df = morph_cat_df.loc[:, morph_cat_df.isna().mean() <= 0.25]
morph_cat_df = morph_cat_df.dropna()
morph_cat_df = morph_cat_df.loc[:, morph_cat_df.std() > 0].astype(np.float64)

#remove all non numerical collumns
morph_cat_df = morph_cat_df.select_dtypes(exclude=['object'])

#run a pca on to determine important features

#exclude insdentifying features
exclude_features = ['OBJECT_ID','RIGHT_ASCENSION','DECLINATION']
morph_features = morph_cat_df.drop(columns=exclude_features, errors = "ignore")

#scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(morph_features)

#run pca
pca_model = PCA(n_components=0.95, random_state=ran_state )
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

#make dataframe of important features and sort
feature_importance_df = pd.DataFrame({
    'Feature': morph_features.columns,
    'Importance': feature_importance
}).sort_values(by="Importance", ascending=False)



#plot feature importance for top n_imp_features features
plt.figure(figsize=(10, 8))

plt.barh(
    feature_importance_df['Feature'].head(n_imp_features)[::-1],
    feature_importance_df['Importance'].head(n_imp_features)[::-1]
)

plt.xlabel('PCA Feature Contribution')
plt.ylabel('Feature')
plt.title('Top Features from PCA')

plt.tight_layout()
plt.savefig('pca_10_imp_feat.png', dpi=200)
plt.close()

#select most important original features
top_features = feature_importance_df['Feature'].head(n_imp_features).tolist()
selected_idx = morph_features.columns.get_indexer(top_features)

X_selected = X_scaled[:, selected_idx]

#train hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples)
labels = clusterer.fit_predict(X_selected)

#evaluate hdbscan
labels_s = pd.Series(labels, index=morph_cat_df.index, name="cluster")

#evaluate hdbscan 

#remove noise from hdbsacm
rm_noise = labels != -1                              
n_clusters = len(set(labels[rm_noise]))
print(f"Clusters: {n_clusters}, noise points: {(~rm_noise).mean():.1%}")

if n_clusters >= 2:
    silhouette_avg = silhouette_score(
        X_selected[rm_noise], labels[rm_noise],
        sample_size=min(10000, int(rm_noise.sum())), random_state=42
    )
    print("Silhouette Coefficient: {:.2f}".format(silhouette_avg))
else:
    print("Silhouette Coeffienct can't be calculated below two clusters")

#visualise clusters with density plot

plt.figure(figsize=(10, 6))

for cluster in labels_s.value_counts().index[:6]:            # largest groups (-1 = noise)
    cluster_data = X_selected[labels == cluster, 0]
    name = "noise" if cluster == -1 else f"Cluster {cluster}"
    plt.hist(cluster_data, bins=40, density=True, histtype="step", lw=1.5,
             label=f"{name} (N={len(cluster_data):,})")

plt.xlabel(f"{top_features[0]} (scaled)")
plt.ylabel("Density")
plt.title("HDBSCAN Cluster Density Distribution")
plt.legend()
plt.savefig("hdbscan_cluster_density.png", dpi=200)