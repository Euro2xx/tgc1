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



# class myposembLayer(tf.keras.layers.Layer):
#     def __init__(self, num_outputs):
#         super(myposembLayer, self).__init__()
#         self.num_outputs = num_outputs
#
#     def build(self, input_shape):
#         self.kernel =
#
#     def call (self, inputs):
#
# class mytokenizer(tf.keras.layers.Layer):
#     def __init__(self):
#         super(mytokenizer, self).__init__()
#         self
#
#     def build(self):
#
#     def call(self):
#         return tf.tokenizer
#
# class mymatmul(tf.keras.layers.Layer):
#     def __init__(self):
#
#     def build(self):
#
#     def call(self):
#         return tf.
#
# class mynorm(tf.keras.layers.Layer):
#     def __init__(self):
#         super(mynorm, self).__init__()
#         self
#
#     def build(self):
#
#     def call(self):
#         return tf.

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)




    class TokenAndPositionEmbedding(layers.Layer):
        def __init__(self, maxlen, vocab_size, embed_dim):
            super(TokenAndPositionEmbedding, self).__init__()
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

        def call(self, x):
            maxlen = tf.shape(x)[-1]
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
            x = self.token_emb(x)
            return x + positions




def Discriminator(config_train):

    # Create the discriminator.
    discriminator = keras.Sequential(
        [
            keras.layers.InputLayer((32, 32, config_train.discriminator_in_channels)),
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.GlobalMaxPooling2D(),
            layers.Dense(1),
        ],
        name="discriminator",
    )



def Generator(config_train):
    generator = keras.Sequential(
        [
            keras.layers.InputLayer((config_train.generator_in_channels,)),

            layers.Dense(8 * 8 * config_train.generator_in_channels),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((8, 8, config_train.generator_in_channels)),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        ],
        name="generator",
    )


