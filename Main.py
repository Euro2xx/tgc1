import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

import statistics
import scipy.misc
import logging
import ntpath
import math
import os
import pickle as pkl
import tarfile
import time
import imageio
from PIL import Image




parser = argparse.ArgumentParser(description="Variables and Configs")
parser.add_argument("--dataset", default="val", choices=["train", "test"])
parser.add_argument("--image_size", default="32", type=int)
parser.add_argument("--num_channels", default="3", type=int)
parser.add_argument("--num_classes", default="10", type=int)
parser.add_argument("--num_layers", default="5", type=int)
parser.add_argument("--train", default="training", choices=["train", "test"])
parser.add_argument("--dropout", default="0.1", type=float)
parser.add_argument("--dim_model", default="64", type=int)
parser.add_argument("--patch_size", default="4", type=int)
parser.add_argument("--latent_dim", default="128", type=int)
parser.add_argument("--weight_decay", default="1e-4", type=float)
parser.add_argument("--lr", default="3e-4", type=float)
parser.add_argument("--num_heads", default="2", type=int)
parser.add_argument("--embed_dim", default="64", type=int)
parser.add_argument("--mlp_dim", default="128", type=int)
parser.add_argument("--noise_dim", default="4096", type=int)
parser.add_argument("--batch_size", default="64", type=int)
parser.add_argument("--generator_in_channels", default="256", type=int)
parser.add_argument("--discriminator_in_channels", default="18", type=int)
parser.add_argument("--layer_norm", default="1e-6", type=float)
parser.add_argument("--projection_dim", default="128", type=int)
parser.add_argument("--num_tokens", default="1", type=int)
parser.add_argument("--filters", default="64", type=int)
parser.add_argument("--num_embeddings", default="128", type=int)
parser.add_argument("--embedding_dim", default="16", type=int)
args = parser.parse_args()

# Dataset from tar file
# def size1(img):
#
#     pixel_val = img
#     Im_val=list(pixel_val.getdata())
#     Im_val_flat=[x for sets in Im_val for x in sets]
    #print(f"Image values {Im_val, Im_val_flat}")


def load_cifar(root, levels=256, with_y=False):
    dataset = 'cifar-10-python.tar.gz'
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(root, dataset)
        if os.path.isfile(new_path) or data_file == 'cifar-10-python.tar.gz':
            dataset = new_path

    f = tarfile.open(dataset, 'r:gz')
    b1 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_1"), encoding="bytes")
    b2 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_2"), encoding="bytes")
    b3 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_3"), encoding="bytes")
    b4 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_4"), encoding="bytes")
    b5 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_5"), encoding="bytes")
    test = pkl.load(f.extractfile("cifar-10-batches-py/test_batch"), encoding="bytes")
    train_x = np.concatenate([b1[b'data'], b2[b'data'], b3[b'data'], b4[b'data'], b5[b'data']], axis=0) / 255.
    train_x = np.asarray(train_x, dtype='float32')
    train_t = np.concatenate([np.array(b1[b'labels']),
                              np.array(b2[b'labels']),
                              np.array(b3[b'labels']),
                              np.array(b4[b'labels']),
                              np.array(b5[b'labels'])], axis=0)

    test_x = test[b'data'] / 255.
    test_x = np.asarray(test_x, dtype='float32')
    test_t = np.array(test[b'labels'])
    f.close()

    train_x = train_x.reshape((train_x.shape[0], 3, 32, 32))
    train_x=train_x.transpose((0,2,3,1))
    test_x = test_x.reshape((test_x.shape[0], 3, 32, 32))
    test_x=test_x.transpose((0,2,3,1))


    # train_x = (train_x, levels) / (levels - 1.)
    # test_x = (test_x, levels) / (levels - 1.)




    # if with_y:
    #     return (train_x, train_t), (test_x, test_t)
    return train_x, test_x


train_x, test_x=load_cifar('Datasets/')

print(f"Training samples: {len(train_x)}")
print(f"Test samples.   {len(test_x)}")
print(f"Training samples: {train_x.shape}")
print(f"Test samples.   {test_x.shape}")


# # Load the CIFAR-10 dataset the keras Dataset.
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# (x_train, y_train), (x_val, y_val) = (
#     (x_train[:40000], y_train[:40000]),
#     (x_train[40000:], y_train[40000:]),
# )
# print(f"Training samples: {len(x_train)}")
# print(f"Validation samples: {len(x_val)}")
# print(f"Testing samples: {len(x_test)}")

# # Convert to tf.data.Dataset objects.
#
# if args.dataset == 'train':
#     x_train = x_train.astype("float32") / 255.0
#     y_train = keras.utils.to_categorical(y_train, 10)
#     train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#     train_ds = train_ds.shuffle(args.batch_size * 100).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
# if args.dataset == 'val':
#     x_val = x_val.astype("float32") / 255.0
#     y_val = keras.utils.to_categorical(y_val, 10)
#     val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
#     val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
# if args.dataset == 'test':
#     x_test = x_test.astype("float32") / 255.0
#     y_test = keras.utils.to_categorical(y_test, 10)
#     test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#     test_ds = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)



def generateimg(img):
    size = list(img)
    fig = plt.figure(figsize=(4, 4))

    for i in range(8):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np.uint8(img[i, :, :, :] * 255))
        plt.axis('off')
    fig.savefig("imges")
    plt.show()



def time1():
    time1=time
    print(time1)
    return time1
def time2():
    time2=time
    print(time2)
    return time2

def time3():
    time3=time
    print(time3)
    return time3
#size of the image

generateimg(train_x)


def show(epoch, trained_gen):
    real_images = train_x[0:5]

    fake = trained_gen.predict(real_images, training=False)

    for i in range(fake.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np.uint8(fake[i, :, :, :] * 255))
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

    return fake
#
# generateimg(x_val[0:5])


# Input architecture
hidden_units = [args.mlp_dim * 2, args.mlp_dim]
num_patches = (args.image_size // args.patch_size) ** 2
print(f"hidden units and num patches {hidden_units, num_patches}")


# Custom Layers and Blocks
# Create Patches Method


class position_embedding(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(position_embedding, self).__init__()
        print(f"Embedding inputs {num_patches, projection_dim}")

        self.encoded_positions = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=num_patches, delta=1)
        encoded_positions = self.encoded_positions(positions)

        return x + encoded_positions


# Data Augmentation

# data_augmentation = keras.Sequential(
#     [
#         layers.Rescaling(1 / 255.0),
#         layers.Resizing(args.image_size + 20, args.image_size + 20),
#         layers.RandomCrop(args.image_size, args.image_size),
#         layers.RandomFlip("horizontal"),
#     ],
#     name="data_augmentation",
# )


def mlp(x, dropout, hidden_units):
    # Iterate over the hidden units and
    # add Dense => Dropout.
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout)(x)
    return x



class project_patches(layers.Layer):
    def __init__(self, mlp_dim, patch_size):
        super(project_patches, self).__init__()

        self.projected_patches = layers.Conv2D(
            filters=mlp_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID", )

    def call(self, x):
        x = self.projected_patches(x)
        _, h, w, c = x.shape
        x = layers.Reshape((h * w, c))(x)
        return x


class attention_map(layers.Layer):
    def __init__(self, num_tokens, layer_norm):
        super(attention_map, self).__init__()
        print(f"inputs of attention map {num_tokens, layer_norm}")
        self.Inputlayer = layers.LayerNormalization(epsilon=layer_norm)
        self.amap = keras.Sequential([
            layers.Conv2D(
                filters=num_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            layers.Conv2D(
                filters=num_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            layers.Conv2D(
                filters=num_tokens,
                kernel_size=(3, 3),
                activation="sigmoid",
                padding="same",
                use_bias=False,
            ),
            layers.Reshape((-1, num_tokens)),
            layers.Permute((2, 1)),
        ])

    def call(self, inputs):
        x = self.Inputlayer(inputs)
        x = self.amap(x)
        num_filters = inputs.shape[-1]
        inputs = layers.Reshape((1, -1, num_filters))(inputs)
        attended_inputs = (
                x[..., tf.newaxis] * inputs
        )
        outputs = tf.reduce_mean(attended_inputs, axis=2)  # (B, num_tokens, C)
        return outputs


class attention_map_comp(layers.Layer):
    def __init__(self, num_tokens, layer_norm):
        super(attention_map_comp, self).__init__()
        print(f"inputs of attention map {num_tokens, layer_norm}")
        self.Inputlayer = layers.LayerNormalization(epsilon=layer_norm)
        self.amap = keras.Sequential([
            layers.Conv2D(
                filters=num_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            layers.Conv2D(
                filters=num_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            layers.Conv2D(
                filters=num_tokens,
                kernel_size=(3, 3),
                activation="sigmoid",
                padding="same",
                use_bias=False,
            ),
            layers.Reshape((-1, num_tokens)),
            layers.Permute((2, 1)),
        ])

    def call(self, inputs):
        x = self.Inputlayer(inputs)
        x = self.amap(x)
        num_filters = inputs.shape[-1]
        inputs = layers.Reshape((1, -1, num_filters))(inputs)
        attended_inputs = (
                x[..., tf.newaxis] * inputs
        )
        outputs = tf.reduce_mean(attended_inputs, axis=2)  # (B, num_tokens, C)
        print(f"Size of the image in bytes {sys.getsizeof(outputs)}")
        return outputs


class transformer(layers.Layer):
    def __init__(self, num_heads, mlp_dim, dropout):
        super(transformer, self).__init__()
        print(f"inputs transformer layer {num_heads, mlp_dim, dropout}")
        self.inputlay = layers.LayerNormalization(epsilon=1e-6)
        self.MHA = layers.MultiHeadAttention(num_heads=num_heads, key_dim=mlp_dim, dropout=dropout)
        self.add = layers.Add()
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
        ])

    def call(self, encoded_patches):
        x1 = self.inputlay(encoded_patches)
        attention_output = self.MHA(x1, x1)
        x2 = self.add([attention_output, x1])
        x3 = self.inputlay(x2)
        x4 = self.mlp(x3)
        encoded_patches = self.add([x2, x4])
        return encoded_patches


class prep_att(layers.Layer):
    def __init__(self):
        super(prep_att, self).__init__()

    def call(self, x):
        y = x.shape

        h = y[1]
        c = y[2]

        h = int(math.sqrt(h))
        x = layers.Reshape((h, h, c))(x)
        return x


class size2(layers.Layer):
    def __init__(self):
        super(size2, self).__init__()

    def call(self, x):
        print(f"Size2 of the image in bytes {sys.getsizeof(x)}")
        # size2=sys.getsizeof(x)
        return x

class size(layers.Layer):
    def __init__(self):
        super(size, self).__init__()


    def call(self, x):

        metric_values = [elem[0] for elem in x]
        file_size_values = [elem[1] for elem in x]
        vmaf_values = [elem[2] for elem in x]
        metric_values=statistics.mean(metric_values)
        file_size_values =statistics.mean(file_size_values)
        length=len(metric_values)
        vmaf_values=statistics.mean(vmaf_values)

        #size1(y)
        print(f"Size of the image in bytes {metric_values,file_size_values,vmaf_values}")

        return x

class timeg(layers.Layer):
    def __init__(self):
        super(timeg, self).__init__()

    def call(self, x):
        print(f"time {time.ctime()}")

        return x
class noise(layers.Layer):
    def __init__(self, latent_dim):
        super(noise, self).__init__()
        self.latent_dim = latent_dim
        # self.one_hot_labels=one_hot_labels

    def call(self, x):
        batch_size = tf.shape(x)[0]


        print(f"get the size {sys.getsizeof(x[1])}")
        random_latent_vector1 = tf.random.normal(shape=(batch_size, 3, self.latent_dim))
        # random_latent_vector1 = random_latent_vector1.astype("uint8")
        x = tf.concat([x, random_latent_vector1], axis=1)
        # x=tf.concat( [x, self.one_hot_labels], axis=1)

        return x


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):

        super(ReflectionPadding2D, self).__init__(**kwargs)

        self.padding = padding

    def call(self, x, mask=None):

        padding_width, padding_height = self.padding

        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(x, padding_tensor, mode="REFLECT")


class residual_block(layers.Layer):
    def __init__(self, activation, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02), kernel_size=(3, 3),
                 strides=(1, 1), gamma_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False, padding ="valid"):
        super(residual_block, self).__init__()

        self.activation=activation
        self.conv=layers.Conv2D(512, kernel_size=kernel_size,strides=strides,kernel_initializer=kernel_initializer,padding=padding,use_bias=use_bias,)
        self.conv1 = layers.Conv2D(512, activation='relu', kernel_size=kernel_size, strides=strides,
                                  kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias, )
        self.add=layers.Add()
        self.Instance=tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)

    def call(self, x):
        dim = x.shape[-1]

        input_tensor=x
        x=  ReflectionPadding2D()(x)
        x=  self.conv(x)
        x=  self.Instance(x)

        x=  self.activation(x)
        x=  ReflectionPadding2D()(x)
        x=  self.conv(x)
        x=  self.Instance(x)
        x=  self.add([input_tensor,x])


        return x




class upsample(layers.Layer):
    def __init__(self, filters, activation, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            gamma_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False):
        super(upsample, self).__init__()


        self.Trans = layers.Conv2DTranspose(filters,kernel_size, strides=strides,padding=padding,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias)
        self.Inst = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)
        self.activation=activation
    def call(self, x):
        x=self.Trans(x)
        x=self.Inst(x)



        x = self.activation(x)
        return x

    # Quantizer

class VectorQuantizer(layers.Layer):
        def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
            super().__init__(**kwargs)
            self.embedding_dim = embedding_dim
            self.num_embeddings = num_embeddings
            self.beta = (
                beta  # This parameter is best kept between [0.25, 2] as per the paper.
            )

            # Initialize the embeddings which we will quantize.
            w_init = tf.random_uniform_initializer()
            self.embeddings = tf.Variable(
                initial_value=w_init(
                    shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
                ),
                trainable=True,
                name="embeddings",
            )

        def call(self, x):
            # Calculate the input shape of the inputs and
            # then flatten the inputs keeping `embedding_dim` intact.
            input_shape = tf.shape(x)
            flattened = tf.reshape(x, [-1, self.embedding_dim])

            # Quantization.
            encoding_indices = self.get_code_indices(flattened)
            encodings = tf.one_hot(encoding_indices, self.num_embeddings)
            quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
            quantized = tf.reshape(quantized, input_shape)

            # Calculate vector quantization loss and add that to the layer. You can learn more
            # about adding losses to different layers here:
            # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
            # the original paper to get a handle on the formulation of the loss function.
            commitment_loss = self.beta * tf.reduce_mean(
                (tf.stop_gradient(quantized) - x) ** 2
            )
            codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
            self.add_loss(commitment_loss + codebook_loss)

            # Straight-through estimator.
            quantized = x + tf.stop_gradient(quantized - x)
            return quantized

        def get_code_indices(self, flattened_inputs):
            # Calculate L2-normalized distance between the inputs and the codes.
            similarity = tf.matmul(flattened_inputs, self.embeddings)
            distances = (
                    tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
                    + tf.reduce_sum(self.embeddings ** 2, axis=0)
                    - 2 * similarity
            )

            # Derive the indices for minimum distances.
            encoding_indices = tf.argmin(distances, axis=1)
            return encoding_indices

# 2. Step Create the generator.


generator = keras.Sequential(
    [
        # Input after data preparation
        timeg(),
        # size(),
        keras.layers.InputLayer(input_shape=(args.image_size, args.image_size, args.num_channels)),
        # layers.BatchNormalization(),
        # Start of the encoding
        # prepare the image

        #data_augmentation,

        # project patches
        project_patches(args.mlp_dim, args.patch_size),

        # add positional embedding to the tokens
        position_embedding(num_patches, args.projection_dim),
        # size(),
        # Dropout for better performance
        layers.Dropout(args.dropout),

        # # Add transfomer Layer
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        layers.LayerNormalization(),
        prep_att(),

        attention_map(args.num_tokens, args.layer_norm),
        #VectorQuantizer(args.num_embeddings, args.embedding_dim),
        noise(args.latent_dim),
        # size(),

        transformer(args.num_heads, args.mlp_dim, args.dropout),
        transformer(args.num_heads, args.mlp_dim, args.dropout),

        transformer(args.num_heads, args.mlp_dim, args.dropout),
        # size(),
        timeg(),

        # rebuilding the image
        layers.Dense(2048, use_bias=False),
        layers.Reshape((4, 4, 512)),
        # ReflectionPadding2D(padding=(3,3)),
        residual_block(activation=layers.Activation("relu")),
        residual_block(activation=layers.Activation("relu")),
        residual_block(activation=layers.Activation("relu")),
        # residual_block(activation=layers.Activation("relu")),
        # residual_block(activation=layers.Activation("relu")),
        # residual_block(activation=layers.Activation("relu")),
        # residual_block(activation=layers.Activation("relu")),
        # residual_block(activation=layers.Activation("relu")),
        # residual_block(activation=layers.Activation("relu")),

        # Upsample

        upsample(128, activation=layers.Activation("relu")),

        upsample(64, activation=layers.Activation("relu")),

        upsample(32, activation=layers.Activation("relu")),
        #

        # Outputlayer
        layers.Conv2D(3, (8, 8), padding="same", activation="sigmoid"),
        timeg(),
        # size(),
    ],
    name="generator",
)

# 3. Step Create the discriminator.
discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((32, 32, 3)),
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        tfa.layers.InstanceNormalization(),

        layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        tfa.layers.InstanceNormalization(),

        layers.Conv2D(256, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        tfa.layers.InstanceNormalization(),

        layers.Conv2D(512, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        tfa.layers.InstanceNormalization(),


        layers.Conv2D(1, (4, 4), strides=1, padding='same'),
    ],
    name="discriminator",
)

#GEt generators

gen_F = generator

# Get the discriminators
disc_X = discriminator




class CycleGan(keras.Model):
    def __init__(self,

        generator_F,
        discriminator_X,

        lambda_cycle=10.0,
        lambda_identity=0.5,):


        super(CycleGan, self).__init__()

        self.gen_F = generator_F
        self.disc_X = discriminator_X

        self.lambda_cycle=lambda_cycle
        self.lambda_identity=lambda_identity
        self.latent_dim = args.latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

        self.mlp_head = keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(args.mlp_dim, activation=tf.nn.gelu),
                layers.Dropout(args.dropout),
                layers.Dense(args.num_classes),
            ]
        )

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(
            self,

            gen_F_optimizer,
            disc_X_optimizer,

            gen_loss_fn,
            disc_loss_fn,
    ):
        super(CycleGan, self).compile()

        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer

        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn

        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, data):
        # Unpack the data.
        real_x= data

        with tf.GradientTape(persistent=True) as tape:
            # Real to fake img
            # fake_y = self.gen_G(real_x, training=True)
            # Fake to real img
            fake_x = self.gen_F(real_x, training=True)



            id_loss = (
                    self.identity_loss_fn(real_x, fake_x)
                    * self.lambda_cycle
                    * self.lambda_identity
            )

            gen_F_loss = id_loss+generator_loss_fn(fake_x)

            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)



            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)

        grads_F = tape.gradient(gen_F_loss, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        # disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        # self.gen_G_optimizer.apply_gradients(
        #     zip(grads_G, self.gen_G.trainable_variables)
        # )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        # self.disc_Y_optimizer.apply_gradients(
        #     zip(disc_Y_grads, self.disc_Y.trainable_variables)
        # )

        return {
            "G_loss": gen_F_loss,

            "D_X_loss": disc_X_loss,

        }
    def call(self, inputs):
        fake =gen_F(inputs)
        pred = disc_X( fake)
        return fake

# class GANMonitor(keras.callbacks.Callback):
#     """A callback to generate and save images after each epoch"""
#
#     def __init__(self, num_img=4):
#         self.num_img = num_img
#
#     def on_epoch_end(self, epoch, logs=None):
#
#             _, ax = plt.subplots(10, 20, figsize=(12, 12))
#             for i in range(4):
#                 img = test_x[0:5]
#                 prediction = cy_gan.gen_F.predict(img)
#                 prediction = (prediction[i] * 255).astype(np.uint8)
#                 img = (img[i]* 255).astype(np.uint8)
#
#                 ax[i, 0].imshow(img[i])
#                 ax[i, 1].imshow(prediction[i])
#                 ax[i, 0].set_title("Input image")
#                 ax[i, 1].set_title("Rebuild image")
#                 ax[i, 0].axis("off")
#                 ax[i, 1].axis("off")
#
#                 prediction = keras.preprocessing.image.array_to_img(prediction)
#                 prediction.save(
#                     "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
#                 )
#             plt.show()
#             plt.close()



adv_loss_fn = keras.losses.MeanSquaredError()

#adv_loss_fn=tf.reduce_mean()

def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    # fake_loss = tf.reduce_mean(tf.square(fake - 1.))
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    # real_loss=tf.reduce_mean(tf.square(real - 1.))
    # fake_loss =tf.reduce_mean(tf.square(fake - 1.))
    return (real_loss + fake_loss) * 0.5

cy_gan = CycleGan(
     generator_F=gen_F, discriminator_X=disc_X)

cy_gan.compile(

    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),

    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,

)

# plotter = GANMonitor()
# checkpoint_filepath = "./logs/train.{epoch:03d}"
# cy_gan_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath)

#checkpoint_filepath = 'logs/models/No_noise'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True)
cy_gan.build(train_x.shape)
cy_gan.fit(

    train_x,
    epochs=100,
    #callbacks=[model_checkpoint_callback]
    # callbacks=[plotter, cy_gan_checkpoint_callback]
)


#Summary

cy_gan.gen_F.summary()

cy_gan.disc_X.summary()



# We first extract the trained generator from our Conditional GAN.
# and the input for the generator

trained_gen=cy_gan.gen_F
trained_gen.save('logs/models/No_noise')
#create 5 fake imgs to compare
fake = trained_gen.predict(train_x[0:5])

generateimg(fake)

#compared the image

x_data=train_x[0:5]
y_data=fake

def _tf_fspecial_gauss(size, sigma=1.5):
    """Function to mimic the 'fspecial' gaussian MATLAB function"""
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def SSIM(img1, img2, k1=0.01, k2=0.02, L=1, window_size=11):
    """
    The function is to calculate the ssim score
    """

    img1 = tf.expand_dims(img1, 0)
    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, 0)
    img2 = tf.expand_dims(img2, -1)

    window = _tf_fspecial_gauss(window_size)

    mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides = [1 ,1, 1, 1], padding = 'VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
    sigma1_2 = tf.nn.conv2d(img1*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2

    c1 = (k1*L)**2
    c2 = (k2*L)**2

    ssim_map = ((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))

    return tf.reduce_mean(ssim_map)

def tf_log10(x):
    numerator = tf.experimental.numpy.log10(x)
    denominator = tf.experimental.numpy.log10(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 255.0
    return 10.0 * tf_log10((max_pixel ** 2) / (tf.reduce_mean(tf.square(y_pred - y_true))))

def compare(x_data, fake):
    img1 = np.array(x_data[1]).astype('float32')
    img2 = np.array(fake[1]).astype('float32')

    img1 = tf.constant(img1)
    img2 = tf.constant(img2)

    _SSIM_ = tf.image.ssim(img1, img2, 1.0)
    _PSNR_ = tf.image.psnr(img1, img2, 255.0)

    rgb1 = tf.unstack(img1, axis=2)
    r1 = rgb1[0]
    g1 = rgb1[1]
    b1 = rgb1[2]

    rgb2 = tf.unstack(img2, axis=2)
    r2 = rgb2[0]
    g2 = rgb2[1]
    b2 = rgb2[2]

    ssim_r=SSIM(r1,r2)
    ssim_g=SSIM(g1,g2)
    ssim_b=SSIM(b1,b2)

    ssim = tf.reduce_mean(ssim_r+ssim_g+ssim_b)/3
    psnr = PSNR(img1, img2)



    print(ssim)
    print(psnr)


compare(x_data, fake)








