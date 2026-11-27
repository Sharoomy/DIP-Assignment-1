#isolation forest

#imports
from sklearn.ensemble import IsolationForest

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from astropy.io import fits
from astropy.table import Table
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#parameters for isolation forest
contamination = 0.05  
n_estimators = 100
sample_size = 256
#load data

path = Path("/user/home/dl23921/ass1/DIP_2026")

morph_cat = "EUC_MER_FINAL-MORPH-CAT_TILE102159484-AE979F_20241026T003045.182961Z_00.00.fits"

with fits.open(path / morph_cat) as hdul:
    morph_cat_table = Table(hdul[1].data)

#valisdation features

val_features = [
    "MAJOR_MERGER",
    "PECULIAR_YES",
    "CLUMP_COUNT_UNUSUAL_ANY_THRESHOLD"
]
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

#select  most important original features
top_features = feature_importance_df['Feature'].head(
    n_imp_features
).tolist()

#extract selected features from scaled dataset
selected_indices = [
    list(morph_cat_df.columns).index(col)
    for col in top_features
]

X_selected = X_scaled[:, selected_indices]

#print checks after cleaning
print("Original catalogue rows:", len(morph_cat_table))
print("Rows after cleaning:", len(morph_cat_df))
print("Remaining missing values:", morph_cat_df.isna().sum().sum())



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
plt.savefig("anomaly_density_isoforest.png", dpi=200)



#valide using the val features

#combines validation features
results_df = pd.DataFrame({
    'is_anomaly_predicted': morph_cat_df["anomaly"],
    'is_merger_actual': val_df["MAJOR_MERGER"],
    'is_peculiar_actual': val_df["PECULIAR_YES"]
})

true_positives = ((results_df['is_anomaly_predicted'] == -1) & 
                  (results_df['is_merger_actual'] == 1)).sum()

total_predicted_anomalies = (results_df['is_anomaly_predicted'] == -1).sum()

precision = true_positives / total_predicted_anomalies if total_predicted_anomalies > 0 else 0

print(f"Validation Results:")
print(f"Total anomalies detected: {total_predicted_anomalies}")
print(f"Successfully caught known mergers: {true_positives}")
print(f"Precision (Scientific Accuracy): {precision:.2%}")