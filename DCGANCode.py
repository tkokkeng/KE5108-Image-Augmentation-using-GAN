#####################################################################
# This application generates images and masks from a trained GAN. It
# enhances the generated images and masks using image morphologiocal
# operations and equalisation. The images and files are saved to file
# (.png).
#####################################################################

import numpy as np
import keras
from keras import layers
from skimage import io
from skimage.morphology import opening
from skimage.morphology import ball
from skimage.morphology import disk
from skimage.exposure import equalize_adapthist
import string
from random import *

#####################################################################
# Parameters
#####################################################################
latent_dim = 128
height = 128
width = 128
channels = 5

#####################################################################
# Generator
#####################################################################
generator_input = keras.Input(shape=(latent_dim,))

# First, transform the input into a 16x16 128-channels feature map
x = layers.Dense(256 * 64 * 64)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((64, 64, 256))(x)

# Then, add a convolution layer
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Upsample to 32x32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

# Few more conv layers
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Produce a 32x32 1-channel feature map
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()

#####################################################################
# Discriminator
#####################################################################
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)

x = layers.Flatten()(x)

# One dropout layer - important trick!
x = layers.Dropout(0.4)(x)

# Classification layer
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

# To stabilize training, we use learning rate decay
# and gradient clipping (by value) in the optimizer.
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

#####################################################################
# GAN
#####################################################################
# Set discriminator weights to non-trainable
# (will only apply to the `gan` model)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0002, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

#####################################################################
# Load Weights from File
#####################################################################
print('Loading GAN model weights ...', '\n')
gan.load_weights('..\data\\gan_model\\gan4500.h5')
print('GAN model weights loaded.', '\n')

#####################################################################
# Generate GAN Images
#####################################################################
print('Generating GAN images ...', '\n')
num_generated_images = 2000
bw_threshold = 230 # threshold to covert gray scale mask to black white

# Sample random points in the latent space
random_latent_vectors = np.random.normal(size=(num_generated_images, latent_dim))

# Decode them to fake images
generated_images = generator.predict(random_latent_vectors)

image_list = []
mask_list = []
for i in range(generated_images.shape[0]):

    # Separate the image and mask
    img_arr = (generated_images[i][:, :, :4] * 255.).astype(np.uint8)
    image_list.append([img_arr])

    mask_arr = (generated_images[i][:, :, 4] * 255.).astype(np.uint8)
    # Set values in mask to 0 or 255 using threshold
    mask_arr[mask_arr > bw_threshold] = 255
    mask_arr[mask_arr < 255] = 0
    mask_list.append([mask_arr])

print('GAN images generated.', '\n')

#####################################################################
# Enhance GAN Images and Masks
#####################################################################
print('Enhancing GAN images ...', '\n')

# Disk and ball radius
radius = 3.5

processed_image_list = []
selem = ball(radius)
for i in image_list:
    opened = opening(i[0], selem)
    equalized = (equalize_adapthist(opened, clip_limit=0.015) * 255.).astype(np.uint)
    alpha = np.expand_dims(i[0][:, :, 3], axis=2)
    equalized = np.concatenate((equalized, alpha), axis=2)
    processed_image_list.append([equalized])

processed_mask_list = []
selem = disk(radius)
for i in mask_list:
    processed_mask_list.append([opening(i[0], selem)])

print('GAN image enhancement completed.', '\n')

#####################################################################
# Save to File GAN Images and Masks
#####################################################################
print('Saving GAN images ...', '\n')

max_char = 12
allchar = string.ascii_letters + string.digits
save_img_dir = '..\data\\final\\gan_images\\'
save_mask_dir = '..\data\\final\\gan_masks\\'

for i in range(len(processed_image_list)):
    a_filename = "".join(choice(allchar) for x in range(max_char))
    io.imsave(save_img_dir + a_filename + '.png', processed_image_list[i][0])
    io.imsave(save_mask_dir + a_filename + '.png', processed_mask_list[i][0])

print('GAN images saved.', '\n')
