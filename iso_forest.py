#isolation forest

#imports
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from astropy.io import fits
from astropy.table import Table
from pathlib import Path

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

morph_cat_table_features = morph_cat_table[numerical_features]

morph_cat_df = morph_cat_table_features.to_pandas()

morph_cat_df = morph_cat_df.apply(pd.to_numeric, errors='coerce')
morph_cat_df = morph_cat_df.replace([np.inf, -np.inf], np.nan)

morph_cat_df = morph_cat_df.dropna()