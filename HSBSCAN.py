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

import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#settings
min_cluster_size = 10
min_samples = 5
n_imp_features = 10

#load data
path = Path("/user/home/dl23921/ass1/DIP_2026")

#use catalog of galaxy morphology
morph_cat = "EUC_MER_FINAL-MORPH-CAT_TILE102159484-AE979F_20241026T003045.182961Z_00.00.fits"

with fits.open(path / morph_cat) as hdul:
    morph_cat_table = Table(hdul[1].data)


#make all values numerical and drop na with rows above 25%
morph_cat_df = morph_cat_table.to_pandas()

morph_cat_df = morph_cat_df.apply(pd.to_numeric, errors='coerce')
morph_cat_df = morph_cat_df.replace([np.inf, -np.inf], np.nan)
morph_cat_df = morph_cat_df.loc[:, morph_cat_df.isna().mean() <= 0.25]

#remove all non numerical collumns
morph_cat_df = morph_cat_df.select_dtypes(exclude=['object'])

#run a pca on to determine important features

#exclude insdentifying features
exclude_features = ['OBJECT_ID','RIGHT_ASCENSION','DECLINATION']
morph_features = morph_cat_df.drop(columns=exclude_features)

#scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(morph_features)

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
    'Feature': morph_features.columns,
    'Importance': feature_importance
})

#sort important features
feature_importance_df = feature_importance_df.sort_values(
    by='Importance',
    ascending=False
)

#plot feature importance for top 20 features
plt.figure(figsize=(10, 8))

plt.barh(
    feature_importance_df['Feature'].head(20)[::-1],
    feature_importance_df['Importance'].head(20)[::-1]
)

plt.xlabel('PCA Feature Contribution')
plt.ylabel('Feature')
plt.title('Top 20 Features from PCA')

plt.tight_layout()
plt.savefig('pca_10_imp_feat.png', dpi=200)
plt.show()

# select 20 most important original features
top_features = feature_importance_df['Feature'].head(
    n_imp_features
).tolist()

# extract selected features from scaled dataset
selected_indices = [
    list(morph_features.columns).index(col)
    for col in top_features
]

X_selected = X_scaled[:, selected_indices]

#train hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples
)

clusterer.fit(X_selected)

labels = clusterer.labels_


#evaluate hdbscan

labels = clusterer.labels_
silhouette_avg = silhouette_score(X_selected, labels)
#ari = adjusted_rand_score(val_df['MAJOR_MERGER'], labels)

#evaluation metrics
print("Silhouette Coefficient: {:.2f}".format(silhouette_avg))
#print("Adjusted Rand Index: {:.2f}".format(ari))

#visualise clusters with density plot

plt.figure(figsize=(10, 6))

for cluster in set(clusterer.labels_):
    cluster_data = X_selected[clusterer.labels_ == cluster, 0]

    plt.hist(cluster_data, bins=40, density=True, alpha=0.5,
             label=f"Cluster {cluster}")

plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Density")
plt.title("HDBSCAN Cluster Density Distribution")
plt.legend()
plt.show()