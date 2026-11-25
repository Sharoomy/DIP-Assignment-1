#imports
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from astropy.io import fits
from astropy.table import Table
from pathlib import Path

from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.coordinates import SkyCoord
import astropy.units as u

#settings
cutout_size = 64

latent_dimensions = 64
epochs = 3
batch_size = 64

#load data

path = Path("/user/home/dl23921/ass1/DIP_2026")

cutout_cat = "EUC_MER_FINAL-CUTOUTS-CAT_TILE102159484-73BCF1_20241025T145359.184204Z_00.00.fits"

with fits.open(path / cutout_cat) as hdul:
    cutout_table = Table(hdul[1].data)

vis_mosaic = "EUC_MER_BGSUB-MOSAIC-VIS_TILE102159484-116033_20241025T015934.885975Z_00.00.fits"


#make cutouts of galaxies from mosaic image using info from galaxy cutouts catalog

def gal_img_gen(mosaic_img, wcs, cutout_table, split=None):

    for i, row in enumerate(cutout_table):

        #get train and test data
        if split == "train" and i % 5 == 0:
            continue

        if split == "test" and i % 5 != 0:
            continue

        #get galaxy position from catalog
        ra = row['RIGHT_ASCENSION']
        dec = row['DECLINATION']

        position = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

        try:
            cutout = Cutout2D(
                mosaic_img,
                position,
                size=(cutout_size, cutout_size),
                wcs=wcs,
                mode='partial',
                fill_value=np.nan
            )

        except NoOverlapError:
            continue

        if np.isfinite(cutout.data).all():

            image = cutout.data.astype(np.float32)

            yield image, np.int64(row['OBJECT_ID'])


def normalise_images(images, galaxy_id):

    image = tf.clip_by_value(images, lower, upper)
    image = (image - lower) / (upper - lower)

    #resize images to 28x28 for autoencoder input
    image = tf.image.resize(image[..., tf.newaxis], (28, 28))

    return image, galaxy_id

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


with fits.open(path / vis_mosaic, memmap=True) as hdul:
    mosaic_img = hdul[0].data
    wcs = WCS(hdul[0].header)

    print(f"Mosaic image shape: {mosaic_img.shape}")

    #estimate normalisation from a sample of images
    sample_images = []

    for image, galaxy_id in gal_img_gen(
        mosaic_img, wcs, cutout_table
    ):

        sample_images.append(image.ravel())

        if len(sample_images) >= 200:
            break

    if len(sample_images) == 0:
        raise ValueError("No valid galaxy cutouts were found.")

    sample_pixels = np.concatenate(sample_images)

    lower = np.percentile(sample_pixels, 1)
    upper = np.percentile(sample_pixels, 99)

    if upper <= lower:
        raise ValueError("Invalid normalisation limits.")

    print("Lower limit:", lower)
    print("Upper limit:", upper)

    #create train and test datasets
    output_signature = (
        tf.TensorSpec(shape=(64, 64), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )

    train_images = tf.data.Dataset.from_generator(
        lambda: gal_img_gen(
            mosaic_img, wcs, cutout_table, split="train"
        ),
        output_signature=output_signature
    )

    test_images = tf.data.Dataset.from_generator(
        lambda: gal_img_gen(
            mosaic_img, wcs, cutout_table, split="test"
        ),
        output_signature=output_signature
    )

    #preprocess datasets
    
    x_train = train_images.map(normalise_images, num_parallel_calls=tf.data.AUTOTUNE) \
                      .map(lambda image, galaxy_id: (image, image)) \
                      .shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    x_test = test_images.map(normalise_images, num_parallel_calls=tf.data.AUTOTUNE) \
                      .map(lambda image, galaxy_id: (image, image)) \
                      .shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    #train autoencoder

    autoencoder = SimpleAutoencoder(latent_dimensions)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    history = autoencoder.fit(
        x_train,
        epochs=epochs,
        shuffle=True,
        validation_data=x_test
    )

    #calculate reconstruction error for anomaly detection
    reconstruction_errors = []
    original_images = []
    reconstructed_images = []

    for images, targets in x_test:

        reconstructed = autoencoder.predict(images, verbose=0)

        errors = tf.reduce_mean(tf.square(images - reconstructed),axis=(1, 2, 3))

        reconstruction_errors.extend(errors.numpy())

        original_images.extend(images.numpy())
        reconstructed_images.extend(reconstructed.numpy())

    reconstruction_errors = np.array(reconstruction_errors)

    print("Reconstruction errors shape:", reconstruction_errors.shape)
    
    #visualise reconstruction error
    plt.figure(figsize=(10, 5))

    plt.hist(reconstruction_errors, bins=50)

    plt.xlabel("Reconstruction error")
    plt.ylabel("Number of galaxies")
    plt.title("Galaxy Reconstruction Errors")

    plt.savefig("recon_error_autoencoder.png", dpi=200)

    #identify anomalies based on reconstruction error

    anomaly_threshold = np.percentile(reconstruction_errors, 95)

    anomaly_indices = np.where(reconstruction_errors > anomaly_threshold)[0]

    normal_errors = reconstruction_errors[reconstruction_errors <= anomaly_threshold]
    anomaly_errors = reconstruction_errors[anomaly_indices]

    #visualise anomalies using reconstruction error density plot
    plt.figure(figsize=(10, 6))

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

    #anomalies for first three images
    original_anomalies = [original_images[i] for i in anomaly_indices[:3]]
    reconstructed_anomalies = [reconstructed_images[i] for i in anomaly_indices[:3]]
    

    # Visualise anomalies
    plt.figure(figsize=(10, 5))

    for i in range(len(original_anomalies)):

        #original image
        plt.subplot(2, 3, i + 1)
        plt.imshow(original_anomalies[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        #reconstructed image
        plt.subplot(2, 3, i + 4)
        plt.imshow(reconstructed_anomalies[i].squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("gal_compare_autoencoder.png", dpi=200)

