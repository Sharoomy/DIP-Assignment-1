#imports

from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path

#load fits files

path = Path("/user/home/dl23921/ass1/DIP_2026")

vis_mosaic = "EUC_MER_BGSUB-MOSAIC-VIS_TILE102159484-116033_20241025T015934.885975Z_00.00.fits"
infra_H = "EUC_MER_BGSUB-MOSAIC-NIR-H_TILE102159484-39CF67_20241024T214009.099906Z_00.00.fits"
infra_J = "EUC_MER_BGSUB-MOSAIC-NIR-J_TILE102159484-C026DB_20241024T213141.416722Z_00.00.fits"
infra_Y = "EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE102159484-A9CCA7_20241024T221227.957725Z_00.00.fits"

with fits.open(path / vis_mosaic) as hdul:
    vis_img = hdul.info()
    wcs = WCS(hdul[0].header)

with fits.open(path / infra_H) as hdul:
    infra_H_img = hdul.info()
    wcs = WCS(hdul[0].header)

with fits.open(path / infra_J) as hdul:
    infra_J_img = hdul.info()
    wcs = WCS(hdul[0].header)

with fits.open(path / infra_Y) as hdul:
    infra_Y_img = hdul.info()
    wcs = WCS(hdul[0].header)

with fits.open(path / vis_mosaic) as hdul:
    mosaic_img = hdul.info()
    wcs = WCS(hdul[0].header)


#set colour scheme
cmap = plt.get_cmap('BuPu').copy()   

#plot data

fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
fig_data = ax.imshow(mosaic_img, origin='lower', cmap=cmap)
ax.set(xlabel='Right Ascension', ylabel='Declination')
ax.set_title('Sky Tile Data')
fig.colorbar(fig_data, ax=ax)
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(16, 5), subplot_kw=dict(projection=wcs))
ax1, ax2, ax3, ax4 = axes

im = ax1.imshow(vis_img, origin='lower', cmap=cmap)
ax1.set(xlabel='Right Ascension', ylabel='Declination')
ax1.set_title('Visual Mosaic')

ax2.imshow(infra_H_img, origin='lower', cmap=cmap)
ax2.set(xlabel='Right Ascension', ylabel='Declination')
ax2.set_title('Infrared H Mosaic')

ax3.imshow(infra_J_img, origin='lower', cmap=cmap)
ax3.set(xlabel='Right Ascension', ylabel='Declination')
ax3.set_title('Infrared J Mosaic')

ax4.imshow(infra_Y_img, origin='lower', cmap=cmap)
ax4.set(xlabel='Right Ascension', ylabel='Declination')
ax4.set_title('Infrared Y Mosaic')

plt.savefig("anomaly_density_isoforest.png", dpi=400)