import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras.models import Sequential
import numpy as np
import argparse
import matplotlib.pyplot as plt
#import os
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
args = parser.parse_args()


#Dataset

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
if args.dataset =='train':
    x_train=x_train.astype("float32") / 255.0
    y_train=keras.utils.to_categorical(y_train, 10)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(args.batch_size * 100).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
if args.dataset=='val':
    x_val = x_val.astype("float32") / 255.0
    y_val = keras.utils.to_categorical(y_val, 10)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
if args.dataset=='test':
    x_test = x_test.astype("float32") / 255.0
    y_test = keras.utils.to_categorical(y_test, 10)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)


def generateimg(pic):
    fig=plt.figure(figsize=(4,4))

    for i in range(5):
        plt.subplot(4,4,i+1)
        plt.imshow(pic[i])
        plt.axis('off')
    plt.show()
#
#generateimg(x_val[0:5])


# Input architecture
hidden_units=[args.mlp_dim*2,args.mlp_dim]
num_patches=(args.image_size//args.patch_size)**2
print(f"hidden units and num patches {hidden_units,num_patches}")


#Custom Layers and Blocks
#Create Patches Method




class position_embedding(layers.Layer):
    def __init__(self,  num_patches, projection_dim):
        super(position_embedding,self).__init__()
        print(f"Embedding inputs { num_patches, projection_dim}")

        self.encoded_positions = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=num_patches, delta=1)
        encoded_positions=self.encoded_positions(positions)



        return x + encoded_positions

#Data Augmentation

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

#get an attention map for further
class project_patches(layers.Layer):
    def __init__(self, mlp_dim, patch_size):
        super(project_patches, self).__init__()

        self.projected_patches  = layers.Conv2D(
        filters=mlp_dim,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
        padding="VALID",)


    def call(self, x):
        x= self.projected_patches(x)
        _, h, w, c = x.shape
        x = layers.Reshape((h * w, c))(x)
        return x

class attention_map(layers.Layer):
    def __init__(self, num_tokens, layer_norm):
        super(attention_map, self).__init__()
        print(f"inputs of attention map {num_tokens, layer_norm}")
        self.Inputlayer=layers.LayerNormalization(epsilon=layer_norm)
        self.amap=keras.Sequential([
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
            layers.Permute((2,1)),
        ])


    def call(self, inputs):
        x=self.Inputlayer(inputs)
        x=self.amap(x)
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
        self.Inputlayer=layers.LayerNormalization(epsilon=layer_norm)
        self.amap=keras.Sequential([
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
            layers.Permute((2,1)),
        ])


    def call(self, inputs):
        x=self.Inputlayer(inputs)
        x=self.amap(x)
        num_filters = inputs.shape[-1]
        inputs = layers.Reshape((1, -1, num_filters))(inputs)
        attended_inputs = (
                x[..., tf.newaxis] * inputs
        )
        outputs = tf.reduce_mean(attended_inputs, axis=2)  # (B, num_tokens, C)
        print(f"Size of the image in bytes {sys.getsizeof(outputs)}")
        return outputs


class transformer(layers.Layer):
    def __init__(self,  num_heads, mlp_dim, dropout):
        super(transformer, self).__init__()
        print(f"inputs transformer layer {num_heads,mlp_dim, dropout}")
        self.inputlay=layers.LayerNormalization(epsilon=1e-6)
        self.MHA=layers.MultiHeadAttention(num_heads=num_heads, key_dim=mlp_dim, dropout=dropout)
        self.add=layers.Add()
        self.mlp= keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
        ])

    def call(self, encoded_patches):
        x1=self.inputlay(encoded_patches)
        attention_output=self.MHA(x1,x1)
        x2=self.add([attention_output,x1])
        x3= self.inputlay(x2)
        x4=self.mlp(x3)
        encoded_patches=self.add([x2,x4])
        return encoded_patches




class prep_att(layers.Layer):
    def __init__(self):
        super(prep_att, self).__init__()

    def call(self, x):
        y=x.shape

        h=y[1]
        c=y[2]

        h=int(math.sqrt(h))
        x=layers.Reshape((h,h,c))(x)
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
        self.latent_dim=latent_dim
        #self.one_hot_labels=one_hot_labels

    def call(self, x):
        batch_size=tf.shape(x)[0]
        random_latent_vector1=tf.random.normal(shape=(batch_size, 4, self.latent_dim))
        #random_latent_vector1 = random_latent_vector1.astype("uint8")
        x=tf.concat([x, random_latent_vector1], axis=1)
        #x=tf.concat( [x, self.one_hot_labels], axis=1)



        return x


class residual(layers.Layer):
    def __init__(self, kernel_size, filters, strides):
        super(noise, self).__init__()
        self.init = tf.contrib.layers.xavier_initializer()
        self.p= int((kernel_size-1)/2)
        self.conv=tf.layers.conv2d( filters=filters, kernel_size=kernel_size, strides=strides,activation=None, padding='VALID')


    def call(self, x):
        p=self.p
        x=tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        x=self.conv(x)
        x= layers.Activation(tf.contrib.layers.instance_norm(x))
        x=tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        x=self.conv(x)
        x



        return x


    # 2. Step Create the generator.
generator = keras.Sequential(
    [
        # Input after data preparation
        keras.layers.InputLayer(input_shape=(32,32,3)),
        #layers.BatchNormalization(),
        #Start of the encoding
        #prepare the image

        data_augmentation,

        #project patches
        project_patches(args.mlp_dim, args.patch_size),

        #add positional embedding to the tokens
        position_embedding(num_patches, args.projection_dim),
        #size(),
        #Dropout for better performance
        layers.Dropout(args.dropout),

        #Add transfomer Layer
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        layers.LayerNormalization(),
        prep_att(),

        attention_map(args.num_tokens, args.layer_norm),

        #size(),
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        transformer(args.num_heads, args.mlp_dim, args.dropout),
        #size(),

        noise(args.latent_dim),

        #rebuilding the image
        layers.Dense(1024, use_bias=False),
        layers.Reshape((4,4,512)),

        # Rebuild with Conv
        layers.Conv2DTranspose(512, (4, 4), strides=(1,1), padding="same", activation='relu'),

        layers.Conv2DTranspose(256, (4, 4), strides=(2,2), padding="same", activation='relu'),
        #layers.BatchNormalization(),
        #layers.Reshape(None, 256, 8, 8),

        layers.Conv2DTranspose(128, (4, 4), strides=(2,2), padding="same", activation="relu"),
        #layers.BatchNormalization(),
        layers.Conv2DTranspose(64, (4, 4), strides=(2,2), padding="same", activation="relu"),
        #layers.BatchNormalization(),
        layers.Conv2DTranspose(32, (4, 4), strides=(2,2), padding="same", activation="relu"),
        #layers.BatchNormalization(),
        layers.Conv2D(32, (4, 4), strides=(2,2), padding="same", activation="relu"),
        # Outputlayer
        layers.Conv2D(3, (8, 8), padding="same", activation="sigmoid"),

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






class ConditionalGAN(keras.Model):
    def __init__(self,  discriminator, generator):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
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

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        # image_one_hot_labels = one_hot_labels[:, :, None, None]
        # print(f"image_one_hot_labels {image_one_hot_labels}")
        # image_one_hot_labels = tf.repeat(
        #     image_one_hot_labels, repeats=[args.image_size*args.image_size*args.num_channels]
        # )
        # print(f"image_one_hot_labels2 {image_one_hot_labels}")
        # image_one_hot_labels = tf.reshape(
        #     image_one_hot_labels, (-1, args.image_size, args.image_size, args.num_classes)
        # )
        # print(f"image_one_hot_labels3 {image_one_hot_labels}")
        # # Sample random points in the latent space and concatenate the labels.
        # # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        # image_one_hot_labels = image_one_hot_labels.astype("float32")

        #input
        random_latent_vectors=  real_images #tf.random.normal(shape=(batch_size, 32,32,3))
        #random_latent_vectors=random_latent_vectors.astype("uint8")
        print(f"random latentvectors  {random_latent_vectors}")
        # one_hot_labels= one_hot_labels.astype("float32")

        # random_vector_labels = tf.concat(
        #     [random_latent_vectors, image_one_hot_labels], axis=3
        # )
        generated_images = self.generator(random_latent_vectors)

        print(f"gemerated images of combined  {generated_images}")



        # fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        # print(f"fake iamges and labels {fake_image_and_labels}")
        # real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        #print(f"real iamges and labels {real_image_and_labels}")
        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        combined_images = tf.concat(
            [generated_images, real_images], axis=0
        )
        print(f"combined images {combined_images}")

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            print(f" shape predictions {predictions.shape}")
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )


        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))



        # Train the generator (note that we should *not* update the weights of the discriminator)!

        with tf.GradientTape() as tape:
            # rebuild images
            fake_images = self.generator(random_latent_vectors)
            print(f"rebuild images {fake_images}")
            #fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_images)
            print(f"predictions2 {predictions}")
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        #print(f"grads as tape {grads.shape}")
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))



        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            

        }

cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

cond_gan.fit(val_ds, epochs=20)

# cond_gan.build(input_shape=(32,32,3))
cond_gan.generator.summary()
cond_gan.discriminator.summary()


# We first extract the trained generator from our Conditional GAN.
#and the input for the generator
trained_gen = cond_gan.generator


def show(data):
    real_images=x_val[0:5]
    print(f"size after attention map {sys.getsizeof(real_images[1])}")
    batch_size = tf.shape(real_images)[0]

    fake = generator.predict(real_images)
    generateimg(fake)
    return fake

show(x_val)




