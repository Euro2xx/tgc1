# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse
import sys

# User-defined

from model import train
from config import config_train, directories

def main(name):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    parser.add_argument("-name", "--name", default="gan-train", help="Checkpoint/Tensorboard label")
    # parser.add_argument("-ds", "--dataset", default="cityscapes", help="choice of training dataset. Currently only supports cityscapes/ADE20k", choices=set(("cityscapes", "ADE20k")), type=str)
    parser.add_argument("--training-data-dir", help="directory holding training images")
    parser.add_argument("--test-data-dir", help="directory holding test images")
    args = parser.parse_args()

    args.name = '{}_train_{}'.format(args.name, time.strftime('%d-%m_%Y_%H_%M'))

    # Launch trainin

    def train(config_train, directories):
        generator_in_channels = config_train.latent_dim + config_train.num_classes
        discriminator_in_channels = config_train.num_channels + config_train.num_classes
        print(generator_in_channels, discriminator_in_channels)
        train(config_train, args)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
