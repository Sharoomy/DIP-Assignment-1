#determine important features
#imports
from os import path
from astropy.io import fits
from astropy.table import Table
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt


#load data

path = ("/user/home/dl23921/ass1/DIP_2026")

#files 
files = {
    "main_cat": "EUC_MER_FINAL-CAT_TILE102159484-EF6D85_20241026T003047.133877Z_00.00.fits",
    "cut_outs": "EUC_MER_FINAL-CUTOUTS-CAT_TILE102159484-73BCF1_20241025T145359.184204Z_00.00.fits",
    "gal_morph": "EUC_MER_FINAL-MORPH-CAT_TILE102159484-AE979F_20241026T003045.182961Z_00.00.fits",
    "gal_sed": "EUC_PHZ_GALAXYSED__20241120T003959.957136Z_00.00.fits",
    "phz_cat": "EUC_PHZ_PHZCAT__20241120T003850.158262Z_00.00.fits",
    "star_sed": "EUC_PHZ_STARSED__20241120T003936.665995Z_00.00.fits"    
}

#load files and print collumn names and first 5 rows
tables = {}


for name, filename in files.items():
    with fits.open(path / filename) as hdul:
        tables[name] = Table(hdul[1].data)
       

hist_pdf = path/"hist_pdf.pdf"

exclude_cols = [
    "OBJECT_ID",
    "SOURCE_ID",
    "ID"
]

with PdfPages(hist_pdf) as pdf:

    for catalogue_name, table in tables.items():

        features = []

        for column in table.colnames:

            #exclude non-numeric columns and columns in the exclude list
            if column in exclude_cols:
                continue

            try:
                values = np.ma.asarray(table[column])

                if values.ndim != 1:
                    continue

                if not np.issubdtype(values.dtype, np.number):
                    continue


                values = values.astype(float).filled(np.nan)
                values = values[np.isfinite(values)]

                if len(values) > 0:
                    features.append((column, values))

            except (TypeError, ValueError):
                continue


        #make histograms of features, 6 per page
        for start in range(0, len(features), 6):

            page_features = features[start:start + 6]

            fig, axes = plt.subplots(3, 2, figsize=(12, 14))
            axes = axes.flatten()

            for ax, (feature, values) in zip(axes, page_features):

                ax.hist(values, bins=50, edgecolor="black")

                ax.set_title(feature)
                ax.set_xlabel("Feature value")
                ax.set_ylabel("Frequency")

            # Remove unused plots
            for ax in axes[len(page_features):]:
                fig.delaxes(ax)

            fig.suptitle(catalogue_name, fontsize=16)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


