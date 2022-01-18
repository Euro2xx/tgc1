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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditions")
    parser.add_argument("--dataset", default="train", choices=["train", "test"])
    parser.add_argument("--train", default="train", choices=["train", "test"])
    parser.add_argument("--dropout", default="0.1")
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    args = parser.parse_args()

    strategy = tf.distribute.MirroredStrategy()
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
    dataset = dataset.shuffle(buffer_size=1024).batch(args.batch_size)

    print(f"Shape of training images: {all_digits.shape}")
    print(f"Shape of training labels: {all_labels.shape}")





    with strategy.scope():

        model = ConditionalGAN(Data_prep=Data_prep,
            discriminator=discriminator, generator=generator, latent_dim=latent_dim,
                    )
        model.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),

            )

    model.fit(dataset, validation = args.dataset, epochs=args.epochs)
# generator.summary()
# discriminator.summary()


# We first extract the trained generator from our Conditional GAN.
    trained_gen = model.generator