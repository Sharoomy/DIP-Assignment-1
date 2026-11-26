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
epochs = 5
batch_size = 64
img_size = 28

#load data

path = Path("/user/home/dl23921/ass1/DIP_2026")

#open cutout catalog
cutout_cat = "EUC_MER_FINAL-CUTOUTS-CAT_TILE102159484-73BCF1_20241025T145359.184204Z_00.00.fits"

with fits.open(path / cutout_cat) as hdul:
    cutout_table = Table(hdul[1].data)

#load visual mosaic
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

        #convert coords into position
        position = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

        #make the cutout
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

        #make sure cutout is valid
        if np.isfinite(cutout.data).all():

            image = cutout.data.astype(np.float32)

            yield image, np.int64(row['OBJECT_ID'])

#resize the images for processing
def normalise_images(images, galaxy_id):

    image = tf.clip_by_value(images, lower, upper)
    image = (image - lower) / (upper - lower)

    #resize images to img_size for autoencoder input
    image = tf.image.resize(image[..., tf.newaxis], (img_size, img_size))

    return image, galaxy_id

#make autoencoder model
class SimpleAutoencoder(Model):

    def __init__(self, latent_dimensions):
        super(SimpleAutoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(img_size, img_size, 1)),
            layers.Flatten(),
            layers.Dense(latent_dimensions, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(img_size * img_size, activation='sigmoid'),
            layers.Reshape((img_size, img_size, 1))
        ])

    def call(self, input_data):
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded

#open the mosaic tile and keep open for all future processing
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

    if len(sample_images) == 0:
        raise ValueError("No cutouts found")

    sample_pixels = np.concatenate(sample_images)

    lower = np.percentile(sample_pixels, 1)
    upper = np.percentile(sample_pixels, 99)

    if upper <= lower:
        raise ValueError("Normalisaton limits not valid")

    print("Lower limit:", lower)
    print("Upper limit:", upper)

    #create train and test datasets
    output_signature = (
        tf.TensorSpec(shape=(cutout_size, cutout_size), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )

    #get images from generator
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

    #preprocess datasets (shuffle training data)
    
    x_train = train_images.map(normalise_images, num_parallel_calls=tf.data.AUTOTUNE) \
                      .map(lambda image, galaxy_id: (image, image)) \
                      .shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    x_test = test_images.map(normalise_images, num_parallel_calls=tf.data.AUTOTUNE) \
                      .map(lambda image, galaxy_id: (image, image)) \
                      .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    #train autoencoder
    autoencoder = SimpleAutoencoder(latent_dimensions)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    history = autoencoder.fit(
        x_train,
        epochs=epochs,
        validation_data=x_test
    )

    #calculate reconstruction error for anomaly detection
    recon_errs = []
    og_imgs = []
    recon_imgs = []

    for images, targets in x_test:

        reconstructed = autoencoder.predict(images, verbose=0)

        errors = tf.reduce_mean(tf.square(images - reconstructed),axis=(1, 2, 3))

        recon_errs.extend(errors.numpy())

        og_imgs.extend(images.numpy())
        recon_imgs.extend(reconstructed)

    recon_errs = np.array(recon_errs)

    print("Reconstruction errors shape:", recon_errs.shape)
    
    #visualise reconstruction error
    plt.figure(figsize=(10, 5))

    plt.hist(recon_errs, bins=50)

    plt.xlabel("Reconstruction error")
    plt.ylabel("Number of galaxies")
    plt.title("Galaxy Reconstruction Errors")

    plt.savefig("recon_error_autoencoder.png", dpi=200)

    #identify anomalies based on reconstruction error

    #set threshold for anomalies
    anomaly_threshold = np.percentile(recon_errs, 95)
    anomaly_indices = np.where(recon_errs > anomaly_threshold)[0]

    #calculate normal and anomaly errors
    normal_errors = recon_errs[recon_errs <= anomaly_threshold]
    anomaly_errors = recon_errs[anomaly_indices]

    #visualise anomalies using reconstruction error density plot
    plt.figure(figsize=(10, 6))

    lo, hi = np.percentile(recon_errs, [0.1, 99.9])
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
    original_anomalies = [og_imgs[i] for i in anomaly_indices[:3]]
    reconstructed_anomalies = [recon_imgs[i] for i in anomaly_indices[:3]]
    

    #compare reconstructed cutout to actual
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

