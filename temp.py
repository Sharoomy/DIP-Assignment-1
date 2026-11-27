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
n_imp_features = 20

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
    list(morph_cat_df.columns).index(col)
    for col in top_features
]

X_selected = X_scaled[:, selected_indices]