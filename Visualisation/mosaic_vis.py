#imports

from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
import numpy as np

#load fits files

path = Path("/user/home/dl23921/ass1/DIP_2026")


files = {
    "vis_mosaic":   "EUC_MER_BGSUB-MOSAIC-VIS_TILE102159484-116033_20241025T015934.885975Z_00.00.fits",
    "infra_H": "EUC_MER_BGSUB-MOSAIC-NIR-H_TILE102159484-39CF67_20241024T214009.099906Z_00.00.fits",
    "infra_J": "EUC_MER_BGSUB-MOSAIC-NIR-J_TILE102159484-C026DB_20241024T213141.416722Z_00.00.fits",
    "infra_Y": "EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE102159484-A9CCA7_20241024T221227.957725Z_00.00.fits",
}

scale = 10
images, wcs_small = {}, {}

for name, fname in files.items():
    with fits.open(path / fname, memmap=True) as hdul:
        images[name] = np.asarray(hdul[0].data[::scale, ::scale], dtype=np.float32)
        wcs_small[name] = WCS(hdul[0].header).slice((slice(None, None, scale), slice(None, None, scale)))

#set colour scheme
cmap = plt.get_cmap('BuPu').copy()   

#plot data

titles = {"vis_mosaic": "Visual Mosaic", "infra_H": "Infrared H Mosaic",
          "infra_J": "Infrared J Mosaic", "infra_Y": "Infrared Y Mosaic"}

fig = plt.figure(figsize=(20, 5.5))
for k, name in enumerate(files, start=1):
    ax = fig.add_subplot(1, 4, k, projection=wcs_small[name])
    img = images[name]
    vmin, vmax = ZScaleInterval().get_limits(img[np.isfinite(img)])
    im = ax.imshow(img, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(xlabel='Right Ascension', ylabel='Declination')
    ax.set_title(titles[name])
    fig.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig("mosaic_visualisation.png", dpi=300)