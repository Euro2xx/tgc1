from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from keras.models import Sequential

from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio
import argparse


from Testenv import ConditionalGAN, Data_prep, discriminator, generator,latent_dim

batch_size=64



(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# Scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels.
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 32, 32, 3))
all_labels = keras.utils.to_categorical(all_labels, 10)

# Create tf.data.Dataset.
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")



parser = argparse.ArgumentParser(description="Conditions")
parser.add_argument("--dataset", default="train", choices=["train", "test"])
parser.add_argument("--image_size", default="32")
parser.add_argument("--train", default="train", choices=["train", "test"])
parser.add_argument("--dropout", default="0.1")
args = parser.parse_args()



cond_gan = ConditionalGAN( Data_prep=Data_prep,
    discriminator=discriminator, generator=generator, latent_dim=latent_dim,
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),

)

cond_gan.fit(dataset, epochs=20)
# generator.summary()
# discriminator.summary()
cond_gan.summary()

# We first extract the trained generator from our Conditional GAN.
trained_gen = cond_gan.generator