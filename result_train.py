import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
import numpy as np
import argparse
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
# import os
# from skimage.util import img_as_ubyte
# from PIL import Image
from tensorflow.python.ops.numpy_ops import np_config
import math

np_config.enable_numpy_behavior()

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
parser.add_argument("--num_tokens", default="4", type=int)
parser.add_argument("--filters", default="64", type=int)
args = parser.parse_args()

# Dataset

# Load the CIFAR-10 dataset.
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
(x_train, y_train), (x_val, y_val) = (
    (x_train[:40000], y_train[:40000]),
    (x_train[40000:], y_train[40000:]),
)
print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Testing samples: {len(x_test)}")
print(f"Size of the image in bytes {sys.getsizeof(x_val)}")
# Convert to tf.data.Dataset objects.

if args.dataset == 'train':
    x_train = x_train.astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(args.batch_size * 100).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
if args.dataset == 'val':
    x_val = x_val.astype("float32") / 255.0
    y_val = keras.utils.to_categorical(y_val, 10)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
if args.dataset == 'test':
    x_test = x_test.astype("float32") / 255.0
    y_test = keras.utils.to_categorical(y_test, 10)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)


def generateimg(pic):
    fig = plt.figure(figsize=(4, 4))

    for i in range(5):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np.uint8(pic[i, :, :, :] * 255))
        plt.axis('off')
    plt.show()


#size of the image

#generateimg(x_val)


def show(epoch, trained_gen):
    real_images = x_val[0:5]

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

data_augmentation = keras.Sequential(
    [
        layers.Rescaling(1 / 255.0),
        layers.Resizing(args.image_size + 20, args.image_size + 20),
        layers.RandomCrop(args.image_size, args.image_size),
        layers.RandomFlip("horizontal"),
    ],
    name="data_augmentation",
)


def mlp(x, dropout, hidden_units):
    # Iterate over the hidden units and
    # add Dense => Dropout.
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout)(x)
    return x


# get an attention map for further
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


class size(layers.Layer):
    def __init__(self):
        super(size, self).__init__()

    def call(self, x):
        print(f"Size of the image in bytes {sys.getsizeof(x)}")

        return x


class noise(layers.Layer):
    def __init__(self, latent_dim):
        super(noise, self).__init__()
        self.latent_dim = latent_dim
        # self.one_hot_labels=one_hot_labels

    def call(self, x):
        batch_size = tf.shape(x)[0]
        print(f"get size of tensor {x.size}")
        print(f"get the size in bytes {sys.getsizeof(x)}")
        print(f"get the size {sys.getsizeof(x[1])}")
        random_latent_vector1 = tf.random.normal(shape=(batch_size, 4, self.latent_dim))
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

    # 2. Step Create the generator.


generator = keras.Sequential(
    [
        # Input after data preparation
        keras.layers.InputLayer(input_shape=(32, 32, 3)),
        # layers.BatchNormalization(),
        # Start of the encoding
        # prepare the image

        data_augmentation,

        # project patches
        project_patches(args.mlp_dim, args.patch_size),

        # add positional embedding to the tokens
        position_embedding(num_patches, args.projection_dim),
        # size(),
        # Dropout for better performance
        layers.Dropout(args.dropout),

        # Add transfomer Layer
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        layers.LayerNormalization(),
        prep_att(),

        attention_map(args.num_tokens, args.layer_norm),

        # size(),
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        # size(),

        noise(args.latent_dim),





        # rebuilding the image
        layers.Dense(1024, use_bias=False),
        layers.Reshape((4, 4, 512)),
        #ReflectionPadding2D(padding=(3,3)),
        residual_block(activation=layers.Activation("relu")),
        residual_block(activation=layers.Activation("relu")),
        residual_block(activation=layers.Activation("relu")),
        residual_block(activation=layers.Activation("relu")),
        # residual_block(activation=layers.Activation("relu")),
        # residual_block(activation=layers.Activation("relu")),
        # residual_block(activation=layers.Activation("relu")),
        # residual_block(activation=layers.Activation("relu")),
        # residual_block(activation=layers.Activation("relu")),






        #Upsample



        upsample(128, activation=layers.Activation("relu")),

        upsample(64, activation=layers.Activation("relu")),

        upsample(32, activation=layers.Activation("relu")),
        #

        # Outputlayer
        layers.Conv2D(3, (7, 7), padding="same", activation="sigmoid"),

    ],
    name="generator",
)

# 3. Step Create the discriminator.
discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((32, 32, 3)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),

        layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),

        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

#GEt generators
gen_G = generator
gen_F = generator

# Get the discriminators
disc_X = discriminator
disc_Y = discriminator



class CycleGan(keras.Model):
    def __init__(self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,):


        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
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
            gen_G_optimizer,
            gen_F_optimizer,
            disc_X_optimizer,
            disc_Y_optimizer,
            gen_loss_fn,
            disc_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, data):
        # Unpack the data.
        real_x, real_y= data

        with tf.GradientTape(persistent=True) as tape:
            # Real to fake img
            fake_y = self.gen_G(real_x, training=True)
            # Fake to real img
            fake_x = self.gen_F(real_x, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_x, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_x, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_x, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                    self.identity_loss_fn(real_x, same_y)
                    * self.lambda_cycle
                    * self.lambda_identity
            )
            id_loss_F = (
                    self.identity_loss_fn(real_x, same_x)
                    * self.lambda_cycle
                    * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }

class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(x_val.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 255).astype(np.uint8)
            img = (img[0] * 255).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = keras.preprocessing.image.array_to_img(prediction)
            prediction.save(
                "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
            )
        plt.show()
        plt.close()


adv_loss_fn = keras.losses.MeanSquaredError()
# adv_loss_fn=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits())

def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5

cy_gan = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y)

cy_gan.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)

plotter = GANMonitor()
checkpoint_filepath = "./logs/train.{epoch:03d}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath)
# cy_gan.build(input_shape=(32,32,3))
cy_gan.fit(
    val_ds,
    epochs=1,

    #callbacks=[plotter, model_checkpoint_callback]
)


#Summary

cy_gan.gen_G.summary()

cy_gan.disc_X.summary()



# We first extract the trained generator from our Conditional GAN.
# and the input for the generator
trained_gen = cy_gan.gen_G
trained_gen2=cy_gan.gen_F




fake = trained_gen.predict(x_val)

generateimg(fake)




