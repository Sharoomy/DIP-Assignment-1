#imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split

from astropy.io import fits
from astropy.table import Table
from pathlib import Path

from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.coordinates import SkyCoord
import astropy.units as u
#clean data

path = Path("/user/home/dl23921/ass1/DIP_2026")

cutout_cat = "EUC_MER_FINAL-CUTOUTS-CAT_TILE102159484-73BCF1_20241025T145359.184204Z_00.00.fits"

with fits.open(path / cutout_cat) as hdul:
    cutout_table = Table(hdul[1].data)

vis_mosaic = "EUC_MER_BGSUB-MOSAIC-VIS_TILE102159484-116033_20241025T015934.885975Z_00.00.fits"




#make cutouts of galaxies from mosaic image using info from galaxy cutouts catalog

cutout_size = 64

galaxy_images = []
galaxy_ids = []

with fits.open(path / vis_mosaic) as hdul:
    hdul.info()
    mosaic_img = np.asarray(hdul[0].data, dtype=np.float32)
    wcs = WCS(hdul[0].header)
    
    for row in cutout_table:

    #get ra and dec of galaxy
        ra = row['RIGHT_ASCENSION']
        dec = row['DECLINATION']

    #convert coordinates to use for mosaic
        position = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

        try:
            #cut out image around galaxy
            cutout = Cutout2D(
                    mosaic_img,
                    position,
                    size=(cutout_size, cutout_size),
                    wcs=wcs,
                    mode='partial',
                    fill_value=np.nan
                )
            print(f"Cutout data shape: {cutout.data.shape}")

        except NoOverlapError:
                continue

        if np.isfinite(cutout.data).all():

            galaxy_images.append(cutout.data)
            galaxy_ids.append(row['OBJECT_ID'])

print(f"Mosaic image type: {type(mosaic_img)}")
print(f"Mosaic image shape: {mosaic_img.shape}")

galaxy_images = np.array(galaxy_images)
galaxy_ids = np.array(galaxy_ids)


#prepare data for autoencoder
images = np.asarray(galaxy_images, dtype=np.float32)

#normalise pixel values 0, 1
lower = np.percentile(images, 1)
upper = np.percentile(images, 99)

images = np.clip(images, lower, upper)
images = (images - lower) / (upper - lower)

#resize images to 28x28 for autoencoder input
images = tf.image.resize(images[..., np.newaxis], (28, 28)).numpy()


x_train, x_test, gal_id_train, gal_id_test = train_test_split(images, galaxy_ids, test_size=0.2, random_state=42)


print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

#make autoencoder model
class SimpleAutoencoder(Model):

    def __init__(self, latent_dimensions):
        super(SimpleAutoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(latent_dimensions, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(28 * 28, activation='sigmoid'),
            layers.Reshape((28, 28, 1))
        ])

    def call(self, input_data):
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded


#train autoencoder

latent_dimensions = 64

autoencoder = SimpleAutoencoder(latent_dimensions)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)


#reconstruct galaxy images

reconstructed = autoencoder.predict(x_test)

#calculate reconstruction error for anomaly detection
reconstruction_errors = np.mean(np.square(x_test - reconstructed), axis=(1, 2, 3))

print(f"Reconstruction errors shape: {reconstruction_errors.shape}")


#visualise reconstruction error
plt.figure(figsize=(10, 5))

plt.hist(reconstruction_errors, bins=50)

plt.xlabel("Reconstruction error")
plt.ylabel("Number of galaxies")
plt.title("Galaxy Reconstruction Errors")

plt.savefig("recon_error_autoencoder.png", dpi=200)



#identify anomalies based on reconstruction error

anomaly_threshold = np.percentile(reconstruction_errors, 95)
anomaly_check = reconstruction_errors > anomaly_threshold

anomaly_indices = np.where(anomaly_check)[0]

normal_errors = reconstruction_errors[~anomaly_check]
anomaly_errors = reconstruction_errors[anomaly_check]

#visualise anomalies using reconstruction error density plot
fig = plt.subplots(figsize=(10, 6))

lo, hi = np.percentile(reconstruction_errors, [0.1, 99.9])
bins = np.linspace(lo, hi, 80)



plt.hist(normal_errors, bins=bins, density=True, alpha=0.5,
        color="tab:green", label="Normal")

plt.hist(anomaly_errors, bins=bins, density=True, alpha=0.5,
        color="tab:red", label="Anomaly")

plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.legend()
plt.savefig("anomaly_density_autoencoder.png", dpi=200)


#compare galaxies with reconstructed images for anomalies

plt.figure(figsize=(10, 5))

for i in range(3):
    idx = anomaly_indices[i]

    #original image
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_test[idx].squeeze(), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    #reconstructed image
    plt.subplot(2, 3, i + 4)
    plt.imshow(reconstructed[idx].squeeze(), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.tight_layout()
plt.savefig("gal_compare_autoencoder.png", dpi=200)

