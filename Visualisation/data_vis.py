#visualising the data

#configs
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

#get rid of warnings
import warnings
from astropy.units import UnitsWarning

warnings.filterwarnings("ignore", category=UnitsWarning)

#imports
from astropy.table import Table
from pathlib import Path

data = Path("/user/home/dl23921/ass1/DIP_2026")

#files (exclude mosaics)
files = {
    "MER_FINAL-CAT": "EUC_MER_FINAL-CAT_TILE102159484-EF6D85_20241026T003047.133877Z_00.00.fits",
    "MER_FINAL-CUTOUTS": "EUC_MER_FINAL-CUTOUTS-CAT_TILE102159484-73BCF1_20241025T145359.184204Z_00.00.fits",
    "MER_FINAL-MORPH-CAT": "EUC_MER_FINAL-MORPH-CAT_TILE102159484-AE979F_20241026T003045.182961Z_00.00.fits",
    "PHZ_GALAXYSED": "EUC_PHZ_GALAXYSED__20241120T003959.957136Z_00.00.fits",
    "PHZ_PHZCAT": "EUC_PHZ_PHZCAT__20241120T003850.158262Z_00.00.fits",
    "PHZ_STARSED": "EUC_PHZ_STARSED__20241120T003936.665995Z_00.00.fits",
    "SPE_WIDE-CAT-LIN": "EUC_SPE_WIDE-CAT-LIN-102159484_N_20241110T054343.637093Z_00.00.fits",
    "SPE_WIDE-CAT-MOD": "EUC_SPE_WIDE-CAT-MOD-102159484_N_20241110T054509.870865Z_00.00.fits",
    "SPE_WIDE-CAT-Z": "EUC_SPE_WIDE-CAT-Z-102159484_N_20241110T054253.801351Z_00.00.fits"
}

#load files and print collumn names and first 5 rows
tables = {}

with open("data_table.txt", "w") as output:

    for name, filename in files.items():

        filepath = data/filename

        print(name, file=output)

        table = Table.read(filepath)

        tables[name] = table

        print("\nColumns:", file=output)
        print(table.colnames, file=output)

        print("\nFirst 5 rows:", file=output)
        print(table[:5], file=output)