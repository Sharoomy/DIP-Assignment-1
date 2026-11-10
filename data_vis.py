#visualising the data
#configs
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Imports
from astropy.table import Table
from pathlib import Path

data = Path("/user/home/dl23921/ass1/DIP_2026")

# FITS files
files = {
    "MER_CAT": "EUC_MER_FINAL-CAT_TILE102159484-EF6D85_20241026T003047.133877Z_00.00.fits",
    "MER_CUTOUTS": "EUC_MER_FINAL-CUTOUTS-CAT_TILE102159484-73BCF1_20241025T145359.184204Z_00.00.fits",
    "MER_MORPH": "EUC_MER_FINAL-MORPH-CAT_TILE102159484-AE979F_20241026T003045.182961Z_00.00.fits",
    "PHZ_GALAXYSED": "EUC_PHZ_GALAXYSED__20241120T003959.957136Z_00.00.fits",
    "PHZ_PHZCAT": "EUC_PHZ_PHZCAT__20241120T003850.158262Z_00.00.fits",
    "PHZ_STARSED": "EUC_PHZ_STARSED__20241120T003936.665995Z_00.00.fits",
    "SPE_LIN": "EUC_SPE_WIDE-CAT-LIN-102159484_N_20241110T054343.637093Z_00.00.fits",
    "SPE_MOD": "EUC_SPE_WIDE-CAT-MOD-102159484_N_20241110T054509.870865Z_00.00.fits",
    "SPE_Z": "EUC_SPE_WIDE-CAT-Z-102159484_N_20241110T054253.801351Z_00.00.fits"
}

# Load and inspect all files
tables = {}

for name, filename in files.items():

    filepath = data / filename

    print(f"\n{'='*70}")
    print(name)
    print(f"{'='*70}")

    table = Table.read(filepath)

    tables[name] = table

    print("\nColumn names:")
    print(table.colnames)

    print("\nFirst 5 rows:")
    print(table[:5])