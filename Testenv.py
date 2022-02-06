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
parser.add_argument("--dataset", default="train", choices=["train", "test"])
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

# Convert to tf.data.Dataset objects.
if args.dataset =='train':
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(args.batch_size * 100).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
if args.dataset=='val':
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
if args.dataset=='test':
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)





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
            #layers.Reshape((-1, num_tokens)),
            #layers.Permute((2,1)),
        ])


    def call(self, inputs):
        x=self.Inputlayer(inputs)
        x=self.amap(x)
        # num_filters = inputs.shape[-1]
        # inputs = layers.Reshape((1, -1, num_filters))(inputs)
        # attended_inputs = (
        #         x[..., tf.newaxis] * inputs
        # )
        # outputs = tf.reduce_mean(attended_inputs, axis=2)  # (B, num_tokens, C)
        return x


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








    # 2. Step Create the generator.
generator = keras.Sequential(
    [
        # Input after data preparation
        keras.layers.InputLayer(input_shape=(4,4,128)),
        #layers.BatchNormalization(),

        layers.Dense(128, use_bias=False),
        layers.Reshape((4,4,128)),

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
        keras.layers.InputLayer((32, 32, 13)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)






class ConditionalGAN(keras.Model):
    def __init__(self, data_augmentation, discriminator, generator, transformer, attention_map,  project_patches, position_embedding):
        super(ConditionalGAN, self).__init__()
        self.data_augmentation=data_augmentation
        self.discriminator = discriminator
        self.generator = generator


        self.transformer=transformer(args.num_heads, args.mlp_dim, args.dropout)

        self.attention_map=attention_map(args.num_tokens, args.layer_norm)
        #self.attention_map_com=attention_map_comp(args.num_tokens, args.layer_norm)
        self.project_patches = project_patches(args.mlp_dim, args.patch_size)
        self.position_embedding =position_embedding(num_patches, args.projection_dim)
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
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[args.image_size * args.image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, args.image_size, args.image_size, args.num_classes)
        )
        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]

        # Encode the images

        x = data_augmentation(real_images)
        print(f"x after data augmentation {x}")
        x=self.project_patches(x)
        print(f"x after project patches {x}")
        x = self.position_embedding(x)
        print(f"x after position embedding  {x}")
        x = layers.Dropout(args.dropout)(x)
        print(f"x after dropout {x}")

        print(f"x after prep{x}")
        for i in range(args.num_layers):
            x=self.transformer(x)
            print(f"x after transformer {x}")
            if i==2:
                _, hh, c = x.shape
                h = int(math.sqrt(hh))
                x = layers.Reshape((h, h, c))(
                    x
                )
                x=self.attention_map(x)
                print(f"attention map x {x}")

        print(f"x after transformer {x}")
        print(f"size after attention map {sys.getsizeof(x)}")
            # x=layers.LayerNormalization(epsilon=args.layer_norm)(x)
            # x=layers.GlobalAveragePooling1D()(x)

        #Get the size of the attention map
        print(f"size after attention map {sys.getsizeof(x)}")
        x= tf.expand_dims(x, axis=1)
        print(f"x after x expansion {x}")
        #create Noise
        #one noise only
        random_latent_vectors2=tf.random.normal(shape=(batch_size,4,4,args.latent_dim))

        #one for concat with the attentionmaps
        random_latent_vectors1 =tf.random.normal(shape=(batch_size,3,4,args.latent_dim))
        print(f"random latent vectors as pure noise {random_latent_vectors1}")

        #combine them with the encoded images
        random_latent_vectors=tf.concat([x,random_latent_vectors1], axis=1)
        print(f"random latent vectors  after combined {random_latent_vectors}")
        # random_vector_labels = tf.concat(
        #     [random_latent_vectors, one_hot_labels], axis=1
        # )
        print(f"random latent vectors {random_latent_vectors}")

        #random_latent_vectors=layers.Reshape((4,4,128))(random_latent_vectors)

        print(f"random latent vectors {random_latent_vectors}")
        generated_images = self.generator(random_latent_vectors2, training=False)
        print(f"gemerated images of combined  {generated_images}")



        image_one_hot_labels=image_one_hot_labels.astype('float32')#
        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        print(f"fake image and labels {fake_image_and_labels}")
        real_images=real_images.astype("float32") / 255.0
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )


        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            print(f" shape predictions {predictions.shape}")
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        #random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # random_vector_labels = tf.concat(
        #     [random_latent_vectors, one_hot_labels], axis=1
        # )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))



        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        # Defining layers



        with tf.GradientTape() as tape:
            # rebuild images
            fake_images = self.generator(random_latent_vectors)
            print(f"rebuild images {fake_images.shape}")
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            print(f"predictions2 {predictions}")
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        print(f"grads as tape ")
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))



        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, data_augmentation=data_augmentation, transformer=transformer, attention_map=attention_map, project_patches=project_patches, position_embedding=position_embedding
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

cond_gan.fit(train_ds, epochs=5)


cond_gan.summary()


# We first extract the trained generator from our Conditional GAN.
# trained_gen = cond_gan.generator