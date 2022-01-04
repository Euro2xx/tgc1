
import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

import data
from data import cifar
import network
from network import CGanf


AUTOTUNE = tf.data.experimental.AUTOTUNE

from config import config_train, directories

def main():

    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--dtype", default="train")
    # parser.add_argument("--image-size", default=32, type=int)
    # parser.add_argument("--patch-size", default=4, type=int)
    # parser.add_argument("--num-layers", default=4, type=int)
    # parser.add_argument("--d-model", default=64, type=int)
    # parser.add_argument("--num-heads", default=4, type=int)
    # parser.add_argument("--mlp-dim", default=128, type=int)
    # parser.add_argument("--lr", default=3e-4, type=float)
    # parser.add_argument("--weight-decay", default=1e-4, type=float)
    # parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    args = parser.parse_args()



    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = network.CGanf(data.cifar(),
            config_train
        )
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            optimizer=tfa.optimizers.AdamW(
                learning_rate=config_train.lr, weight_decay=config_train.weight_decay
            )
        )

    model.fit(
        data.cifar(),
        validation_data=data.cifar(),
        epochs=args.epochs,
        callbacks=[TensorBoard(log_dir=args.logdir, profile_batch=0), ],
    )
    model.save_weights(os.path.join(args.logdir, "vit"))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
