from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
#from keras.models import Sequential

from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio
import argparse


#Variables and Configs
num_channels = 3
num_classes = 10
image_size = 32
latent_dim = 128
embed_dim = 32
num_heads = 2
ff_dim = 32
dropout = 0.1
patch_size = 4
num_layers= 4
mlp_dim = 128
weight_decay = 1e-4
lr = 3e-4
max_len = 1024
batch_size=64


parser = argparse.ArgumentParser(description="Conditions")
parser.add_argument("--dataset", default="train", choices=["train", "test"])
parser.add_argument("--image_size", default="32")
parser.add_argument("--train", default="train", choices=["train", "test"])
parser.add_argument("--dropout", default="0.1")
parser.add_argument("--d-model", default=64, type=int)
args = parser.parse_args()

#Dataset

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





# Input channels for G and D
generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)
Batch_size=all_digits[0]

#Custom Layers and Blocks
#Create Patches Method





# Tokenization latent emb and Pos Emb
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, patches):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = self.add_weight("token_emb", shape=(1, 1, self.d_model))
        self.pos_emb = self.add_weight("pos_emb", shape=(1, self.num_patches + 1, self.d_model))



    def call(self, patches):

        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        positions = self.pos_emb(positions)
        token_emb = self.token_emb(patches)
        data_prep = token_emb + positions
        return data_prep


# MultiHeadSelfAttention
class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=2):
        super(MultiHeadSelfAttention, self).__init__()
        print(f"Multiheadattention inputs {embed_dim, num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = keras.layers.Dense(embed_dim)
        self.key_dense = keras.layers.Dense(embed_dim)
        self.value_dense = keras.layers.Dense(embed_dim)
        self.combine_heads = keras.layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        print(f"Output  Multiheadselfattention {output}")
        return output


# Transformerblock
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super(TransformerBlock, self).__init__()
        print(embed_dim, num_heads, mlp_dim, dropout)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = keras.Sequential(
            [
                keras.layers.Dense(mlp_dim, activation=tfa.activations.gelu),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(embed_dim),
                keras.layers.Dropout(dropout),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        print(f"Output Transformerblock {mlp_output, out1}")
        return mlp_output + out1


# class AttConv2dtranspose(layers.layer):             #opt


# class AttConv2d(layers.layer): #opt


def extract_patches(real_images, image_size, patch_size, num_channels):

    patch_dim = num_channels * patch_size ** 2
    batch_size = tf.shape(real_images)[0]
    patches = tf.image.extract_patches(
            images=real_images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
    patches = tf.reshape(patches, [batch_size, -1, patch_dim])
    print(f"end of Patches {patches}")
    return patches


# 1. Step Preparation(Extract patches, Tokenize, Flatten, Concanate it with Pos Embedding)
class Data_prep(layers.Layer):
    def __init__(self,  patches, mlp_dim, embed_dim,  num_heads, dropout):
        super(Data_prep, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2


        #self.TokenAndPositionEmbedding = TokenAndPositionEmbedding(embed_dim, patches)
        self.TransformerBlock=TransformerBlock(mlp_dim, embed_dim,  num_heads, dropout)
        #print(f"patches{patches.shape}")
        self.Flatten = keras.Sequential([
                        layers.Flatten(),
                        layers.Dense(64, activation='relu')
                        ])







    def call(self, patches, mlp_dim, embed_dim,  num_heads, dropout):
        #print(f"real im images  {real_images.shape}")
        print(f"patches in Dataprep {patches}")
        #x = self.Flatten(patches)
        #print(f"Shape of prepared data after flatten: {x.shape}")
        x = self.TokenAndPositionEmbedding(patches, self.num_patches, self.d_model)
        print(f"Shape of prepared data after token and posemb: {x.shape}")
        x= self.TransformerBlock(mlp_dim, embed_dim, num_heads, dropout)
        print(f"Shape of prepared data after first transblock: {x.shape}")
        x= self.TransformerBlock(mlp_dim, embed_dim, num_heads, dropout)(x)
        x= self.TransformerBlock(mlp_dim, embed_dim, num_heads, dropout)(x)
        return x


# 2. Step Create the generator.
generator = keras.Sequential(
    [
        # Input after data preparation
        keras.layers.InputLayer(input_shape=(138,)),
        #layers.BatchNormalization(),

        layers.Dense(4*4*512, use_bias=False),
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






class ConditionalGAN(keras.Model):
    def __init__(self, Data_prep, discriminator, generator, latent_dim, d_model):
        super(ConditionalGAN, self).__init__()
        self.Data_prep = Data_prep
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.d_model = d_model
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
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        #batch_size = tf.shape(all_digits)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        #
        # 1.Step Data Preparation
        patches=extract_patches(real_images, image_size, patch_size, num_channels)
        print(f"Patches {patches}")
        x = self.Data_prep(patches, mlp_dim, embed_dim,  num_heads, dropout)
        print(f"Shape after preparation{x}")

        #2.Step Concanate image information with random vector

        x = tf.concat([random_vector_labels, x], -1)

        print(f"Shape of prepared data: {x.shape}")

        # 3.Step Rebuild the image
        generated_images = self.generator(x)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # 3. Step  Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

cond_gan = ConditionalGAN( Data_prep=Data_prep,
    discriminator=discriminator, generator=generator, latent_dim=latent_dim, d_model=64
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
