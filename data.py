import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import train
from config import config_test, directories, config_train

# def mnist():
#     (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#     all_digits = np.concatenate([x_train, x_test])
#     all_labels = np.concatenate([y_train, y_test])
#
#     # Scale the pixel values to [0, 1] range, add a channel dimension to
#     # the images, and one-hot encode the labels.
#     all_digits = all_digits.astype("float32") / 255.0
#     all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
#     all_labels = keras.utils.to_categorical(all_labels, 10)
#
#     # Create tf.data.Dataset.
#     dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
#     dataset = dataset.shuffle(buffer_size=1024).batch(config_train.batch_size)
#
#     print(f"Shape of training images: {all_digits.shape}")
#     print(f"Shape of training labels: {all_labels.shape}")


def cifar():

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    #all_digits = np.concatenate([x_train, x_test])
    #all_labels = np.concatenate([y_train, y_test])

    # Optional normalization
    x_train = x_train.astype("float32") / 255.0
    all_digits_train = np.reshape(x_train, (-1, 32, 32, 3))
    all_labels_train = keras.utils.to_categorical(y_train, 10)

    x_test = x_test.astype("float32") / 255.0
    all_digits_test = np.reshape(x_test, (-1, 32, 32, 3))
    all_labels_test = keras.utils.to_categorical(y_test, 10)

    # Create tf.data.Dataset.
    dataset_train = tf.data.Dataset.from_tensor_slices((all_digits_train, all_labels_train))
    dataset_test= tf.data.Dataset.from_tensor_slices((all_digits_test, all_labels_test))
    #dataset = dataset.shuffle(buffer_size=1024).batch(config_train.batch_size)

    print(f"Shape of training images: {all_digits_train.shape}")
    print(f"Shape of training labels: {all_labels_train.shape}")

    if train:
        return dataset_train


    else:
         return dataset_test
