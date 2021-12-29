from tensorflow import keras
from tensorflow.keras import layers

#from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio
import time, os, sys
import argparse
import sys

from config import config_train, directories



class myposembLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(myposembLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel =

    def call (self, inputs):

class mytokenizer(tf.keras.layers.Layer):
    def __init__(self):
        super(mytokenizer, self).__init__()
        self

    def build(self):

    def call(self):
        return tf.tokenizer

class mymatmul




def Discriminator(config_train, data_cifar):

    # Create the discriminator.
    discriminator = keras.Sequential(
        [
            keras.layers.InputLayer((32, 32, discriminator_in_channels)),
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.GlobalMaxPooling2D(),
            layers.Dense(1),
        ],
        name="discriminator",
    )



def Generator(config_train, data_cifar):
    generator = keras.Sequential(
        [
            keras.layers.InputLayer((generator_in_channels,)),
            # We want to generate 128 + num_classes coefficients to reshape into a
            # 7x7x(128 + num_classes) map.
            layers.Dense(7 * 7 * generator_in_channels),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((7, 7, generator_in_channels)),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        ],
        name="generator",
    )


