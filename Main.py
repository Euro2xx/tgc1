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

np_config.enable_numpy_behavior()


parser = argparse.ArgumentParser(description="Variables and Configs")
parser.add_argument("--dataset", default="train", choices=["train", "test"])
parser.add_argument("--image_size", default="32", type=int)
parser.add_argument("--num_channels", default="3", type=int)
parser.add_argument("--num_classes", default="10", type=int)
parser.add_argument("--num_layers", default="3", type=int)
parser.add_argument("--train", default="training", choices=["train", "test"])
parser.add_argument("--dropout", default="0.1", type=float)
parser.add_argument("--dim_model", default=64, type=int)
parser.add_argument("--patch_size", default=4, type=int)
parser.add_argument("--latent_dim", default=128, type=int)
parser.add_argument("--weight_decay", default="1e-4", type=float)
parser.add_argument("--lr", default="3e-4", type=float)
parser.add_argument("--num_heads", default=2, type=int)
parser.add_argument("--embed_dim", default=64, type=int)
parser.add_argument("--mlp_dim", default=128, type=int)
parser.add_argument("--noise_dim", default=4096, type=int)
parser.add_argument("--batch_size", default="64", type=int)
parser.add_argument("--generator_in_channels", default=256, type=int)
parser.add_argument("--discriminator_in_channels", default=18, type=int)

args = parser.parse_args()

#Loading the Dataset
#Loading it with Internet
#args Internet
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

#small subset for computation purposes
pic         =   x_train[np.isin(y_train, [0,1]).flatten()]
pic_labels  =   y_train[np.isin(y_train, [0,1]).flatten()]


sdataset=tf.data.Dataset.from_tensor_slices((pic,pic_labels))
sdataset = sdataset.shuffle(buffer_size=1024).batch(args.batch_size)

pic = tf.convert_to_tensor(pic)

print(pic.shape)
#size of one image in bytes

print(f"Size of the image in bytes {sys.getsizeof(all_digits[1])}")
#show the images



def generateimg(pic):
    fig=plt.figure(figsize=(4,4))

    for i in range(5):
        plt.subplot(4,4,i+1)
        plt.imshow(pic[i])
        plt.axis('off')
    plt.show()

#generate the images of the set

generateimg(pic)
#normalize it
pic = pic.astype("float32") / 255.0
print(f"pic after normalizing {pic.shape}")
#print(f"shape of the small set of images: {pic}")

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")

#Loading it without the Internet
#args







#Custom Layers


#Upsampling the pixels

def pixel_upsample(x, H, W):
    B, N, C = x.shape
    assert N == H*W
    x = tf.reshape(x, (-1, H, W, C))
    x = tf.nn.depth_to_space(x, 2, data_format='NHWC')
    B, H, W, C = x.shape
    x = tf.reshape(x, (-1, H * W, C))
    return x, H, W


#custom noramlization

def normalize_2nd_moment(x, axis=1, eps=1e-8):
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + eps)


#Scaling dot product

def scaled_dot_product(q, k, v):
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_qk = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
    attn_weights = tf.nn.softmax(scaled_qk, axis=-1)
    output = tf.matmul(attn_weights, v)
    return output



# Might use gelu for later layers

B = tf.keras.backend

def gelu(x):
    return 0.5 * x * (1 + B.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


# def generate_and_save_images(generator, epoch, combined, direct, image_size, f_size=2.88):
#     predictions = generator(combined, training=False)
#
#     gen_img = tf.clip_by_value(predictions[0] * 127.5 + 127.5, 0.0, 255.0)
#     gen_img = tf.cast(gen_img, tf.uint8)
#
#     fig = plt.figure(figsize=(f_size, f_size))
#
#     for i in range(gen_img.shape[0]):
#         plt.subplot(8, 8, i + 1)
#         plt.imshow(gen_img[i, :, :, :])
#         plt.axis('off')
#     plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
#
#     path = os.path.join(, '{:04d}.png'.format(epoch))
#     plt.savefig(path)
#
#     # Clear the current axes.
#     plt.cla()
#     # Clear the current figure.
#     plt.clf()
#     # Closes all the figure windows.
#     plt.close('all')





#Turn patches into Tokens and get positional Embeddings (https://proceedings.neurips.cc/paper/2021/file/7c220a2091c26a7f5e9f1cfb099511e3-Paper.pdf)

class Embedding(layers.Layer):
    def __init__(self, image_size, patch_size, dim_model, num_channels, initializer = 'glorot_uniform'):
        super(Embedding,self).__init__()
        print(f"Embedding inputs {image_size, patch_size, dim_model,num_channels}")

        self.num_patches=(image_size//patch_size)**2
        self.patch_dim=num_channels*(patch_size**2)
        self.dim_model= dim_model
        self.token_emb =layers.Embedding(input_dim =self.num_patches, output_dim=self.dim_model, embeddings_initializer=initializer)
        self.pos_emb =layers.Embedding(input_dim=self.num_patches, output_dim=self.dim_model, embeddings_initializer=initializer)
        self.flatten =layers.Flatten()

    def call(self, x):

        x = self.flatten(x)
        pos = tf.range(start=0, limit=self.dim_model, delta=1)
        pos=self.pos_emb(pos)
        tokens = self.token_emb(x)

        #TaP=tf.concat(tokens, pos)

        return tokens+pos





#Preparation for better robustness  (https://arxiv.org/pdf/2111.08413.pdf)

class Prep(layers.Layer):
    def __init__(self):
        super(Prep,self).__init__()
        print(f"Inputs Preparation ")

        self.prenorm=layers.LayerNormalization()
        self.preflatten=layers.Flatten()
    def call(self, x):
        x=self.prenorm(x)
        x=self.preflatten(x)
        #x=x.astype(np.uint8)

        return x

# VIt Layers


class MultiHeadAttention(layers.Layer):
    def __init__(self, dim_model, num_heads, initializer='glorot_uniform'):
        super(MultiHeadAttention,self).__init__()
        print(f"MHA Inputs dim num {dim_model, num_heads}")
        self.num_heads=num_heads
        self.dim_model=dim_model
        assert self.dim_model%self.num_heads ==0

        self.depth = self.dim_model //self.num_heads
        self.wq = layers.Dense(dim_model, kernel_initializer=initializer)
        self.wk = layers.Dense(dim_model, kernel_initializer=initializer)
        self.wv = layers.Dense(dim_model, kernel_initializer=initializer)
        self.dense = layers.Dense(dim_model, kernel_initializer=initializer)

        def split_heads(self,x,batch_size):
            x= tf.reshape(x, (batch_size,-1,self.num_heads, self.depth))
            return  tf.transpose(x, perm=[0,2,1,3])

        def call(self, q,k,v):
            batch_size=tf.shape(q)[0]
            q=self.wq(q)
            k=self.wk(k)
            v=self.wv(v)

            q = self.split_into_heads(q, batch_size)
            k = self.split_into_heads(k, batch_size)
            v = self.split_into_heads(v, batch_size)
            scaled_attention = scaled_dot_product(q, k, v)

            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            original_size_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dim_model))

            output = self.dense(original_size_attention)
            return output

            # MultiHeadSelfAttention
class MultiHeadSelfAttention(layers.Layer):
                def __init__(self, embed_dim, num_heads):
                    super(MultiHeadSelfAttention, self).__init__()

                    print(f"Multiheadattention inputs {embed_dim, num_heads}")
                    if embed_dim % num_heads != 0:
                        raise ValueError(
                            f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
                        )
                    self.num_heads = num_heads
                    self.projection_dim = embed_dim // num_heads
                    self.query_dense = layers.Dense(embed_dim)
                    self.key_dense = layers.Dense(embed_dim)
                    self.value_dense = layers.Dense(embed_dim)
                    self.combine_heads = layers.Dense(embed_dim)

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

                def call(self, x, embed_dim):
                    batch_size = tf.shape(x)[0]
                    query = self.query_dense(x)
                    key = self.key_dense(x)
                    value = self.value_dense(x)
                    query = self.separate_heads(query, batch_size)
                    key = self.separate_heads(key, batch_size)
                    value = self.separate_heads(value, batch_size)

                    attention, weights = self.attention(query, key, value)
                    attention = tf.transpose(attention, perm=[0, 2, 1, 3])
                    concat_attention = tf.reshape(
                        attention, (batch_size, -1, embed_dim)
                    )
                    output = self.combine_heads(concat_attention)
                    return output




# Transformerblock
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super(TransformerBlock, self).__init__()
        print(embed_dim,   num_heads, mlp_dim, dropout)
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.mlp_dim= mlp_dim
        self.dropout=dropout
        self.att = MultiHeadSelfAttention(self.embed_dim, self.num_heads)
        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(self.mlp_dim, activation='relu'),
                layers.Dropout(self.dropout),
                layers.Dense(self.embed_dim),
                layers.Dropout(self.dropout),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout)
        self.dropout2 = layers.Dropout(self.dropout)

    def call(self, x, training):
        inputs_norm = self.layernorm1(x)
        attn_output = self.att(inputs_norm, self.embed_dim)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + x

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1

class Combine(layers.Layer):
    def __init__(self, noise_dim):
        super(Combine, self).__init__()
        self.random_noise=tf.random.normal([1, noise_dim])
        print(f"random noise in combine {self.random_noise}")

    def call(self,x):

        x=x+self.random_noise

        return x

#Generator
#Encoder

class generator(layers.Layer):
    def __init__(self, dim_model, embed_dim, num_heads, mlp_dim, dropout,  noise_dim, modelinitializer ='glorot_uniform'):
        super(generator, self).__init__()
        print(f"inputs generator {dim_model, embed_dim, num_heads, mlp_dim, dropout}")
        model_dim = [512, 256, 128, 64, 32]
        self.dim_model = dim_model
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim=mlp_dim
        self.dropout=dropout
        self.Combine=Combine(noise_dim)
        self.Input=layers.InputLayer(input_shape=(None,64,64))
        self.D=layers.Dense(4*4*256, use_bias=False)
        self.convBlock1=tf.keras.Sequential([
                        layers.Conv2DTranspose(model_dim[2], (4,4), strides=(2,2), padding='same', use_bias=False ),
                        layers.BatchNormalization(),
                        layers.LeakyReLU()

                ])

        self.convBlock2 = tf.keras.Sequential([
            layers.Conv2DTranspose(model_dim[3], (4, 4), strides=(2, 2), padding='same', use_bias=False, input_shape=(8,8,128)),
            layers.BatchNormalization(),
            layers.LeakyReLU()
        ])
        self.convBlock3 = tf.keras.Sequential([
            layers.Conv2DTranspose(model_dim[4], (4, 4), strides=(2, 2),padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU()

        ])
        self.convBlock4 = tf.keras.Sequential([
            layers.Conv2DTranspose(model_dim[4], (4, 4), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU()

            ])
        self.Output=layers.Conv2D(3, (4, 4), strides=(1, 1), padding='same', use_bias=False, activation='sigmoid')
        self.Transformerblock=TransformerBlock(self.embed_dim, self.num_heads, self.mlp_dim, self.dropout)
        self.reshape=layers.Reshape((64,64))
        self.reshape1=layers.Reshape((4,4,256))

    def call(self, x):
        x   =   self.Input(x)

        x   =   self.D(x)
        # Combine the the latent vectors
        x = self.Combine(x)
        x = self.reshape(x)

        # x=layers.BatchNormalization(x)
        #TransformerBlocks
        x   =   self.Transformerblock(x)
        x   =   self.Transformerblock(x)
        x   =   self.Transformerblock(x)


        #Reshape it to image dim
        x   =   self.reshape1(x)
        x=self.convBlock1(x)
        # x=layers.Reshape((x),(8,8,128))
        x=self.convBlock2(x)
        x=self.convBlock3(x)
        #x=self.convBlock4(x)

        x = self.Output(x)
        # x=layers.Reshape((x),(32,32,3))
        x=x.astype(np.uint8)
        return x


class discriminator(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super(discriminator, self).__init__()
        disc_dim = [64, 128, 256, 512]

        self.embed_dim=embed_dim
        self.mlp_dim=mlp_dim
        self.num_heads= num_heads
        self.dropout=dropout


        self.inputlay=layers.InputLayer(input_shape=(None,32,32,3))
        self.block1=tf.keras.Sequential([
            layers.Conv2D(disc_dim[0], (4,4), strides=(2,2), padding ='same', input_shape=(32,32,3)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            ])
        self.block2 = tf.keras.Sequential([
            layers.Conv2D(disc_dim[1], (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
         ])
        self.block3 = tf.keras.Sequential([
            layers.Conv2D(disc_dim[2], (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
        ])
        self.block4 = tf.keras.Sequential  ([
            layers.Conv2D(disc_dim[3], (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
        ])
        self.Outblock =tf.keras.Sequential([
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(1,activation='sigmoid')
        ])
        self.Transformerblock = TransformerBlock(self.embed_dim, self.num_heads, self.mlp_dim, self.dropout)

    def call(self, x):
        x   =   self.inputlay(x)
        #x=      self.Transformerblock(x)
        x   =   self.block1(x)
        x   =   self.block2(x)
        x   =   self.block3(x)
        x   =   self.block4(x)
        x   =   self.Outblock(x)
        return x


class Compression(keras.Model):
    def __init__(self):
        super(Compression,self).__init__()
        self.generator = generator(args.dim_model, args.embed_dim, args.num_heads, args.mlp_dim, args.dropout,  args.noise_dim)
        self.discriminator=discriminator(args.embed_dim, args.num_heads, args.mlp_dim, args.dropout)
        self.embedding = Embedding(args.image_size, args.patch_size, args.dim_model, args.num_channels)
        self.prep = Prep()


        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.mlp_head = keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(args.mlp_dim, activation='relu'),
                layers.Dropout(args.dropout),
                layers.Dense(args.num_classes),
            ]
        )
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")


    @property

    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(Compression, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def extract_patches(self, data, num_channels, patch_size):
        patch_dim = num_channels * patch_size ** 2
        batch_size = tf.shape(data)[0]
        patches = tf.image.extract_patches(
                images=data,
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, patch_dim])
        print(f"end of Patches {patches}")
        return patches

    def train_step(self, data):
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







        #Generator =generator(args.dim_model, args.embed_dim, args.num_heads, args.mlp_dim, args.dropout,  args.noise_dim)
        #Discriminator = discriminator(args.embed_dim, args.num_heads, args.mlp_dim, args.dropout)

        #Encode the images

        # print(f"x  as patches {x}")
        patches=self.extract_patches(self, real_images, args.num_channels, args.patch_size)
        x = self.embedding(patches)
        print(f"x as Token and Positions {x}")
        x=self.prep(x)
        print(f"x after prep {x}")
        print(f"size after prep {sys.getsizeof(x)}")
        gen_images=self.generator(x)

        print(f"gen_images {gen_images}")

        batch_size=tf.shape(real_images)[0]

        print(f"image_one_hot {image_one_hot_labels}")
            # Combine them with real images. Note that we are concatenating the labels
            # with these images here.
        fake_image_and_labels = tf.concat([gen_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
                [fake_image_and_labels, real_image_and_labels], axis=0
            )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )




        print(f"combined after concat {combined_images}")
        #plt.imshow(combined_images[0, :, :, :] * 255)
        print(f"gen_images {gen_images}")

        batch_size=tf.shape(real_images)[0]
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)


        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
             )

        misleading_labels = tf.zeros((batch_size, 1))


        # Train the generator for the images
        with tf.GradientTape() as tape:
            rebuild_images =self.generator(combined_images)
            rebuild_images_and_labels=tf.concat([rebuild_images, image_one_hot_labels],-1)
            predictions=self.discriminator(rebuild_images_and_labels)

            g_loss = self.loss_fn(misleading_labels, predictions)

            # # Produce images for the GIF as you go
            # display.clear_output(wait=True)
            # generate_and_save_images(generator,epoch + 1,seed)
            #
            # # Save the model every 15 epochs
            # if (epoch + 1) % 15 == 0:
            #     checkpoint.save(file_prefix=checkpoint_prefix)


        grads = tape.gradient(self.g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        return {

             "g_loss": self.gen_loss_tracker.result(),
             "d_loss": self.disc_loss_tracker.result(),
         }

    # #
    def call(self, data, training=False):
        # You don't need this method for training.
        # So just pass.
        x=self.extract_patches(data, args.num_channels, args.patch_size)
        x=self.embedding(x)
        x=self.prep(x)
        x=self.generator(x)
        predictions=self.discriminator(x)

        x=self.generator(x)
        predictions=self.discriminator(x)
        return x








model = Compression()
input_shape=(None,32,32,3)
model(sdataset)
model.build(input_shape)
model.summary()
model.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
                 )


#compress.build(input_shape=(10000,32,32,3))
#output = compress((32,32,3))





model.fit(sdataset, epochs=5)

#trained_gen =  compress.generator






