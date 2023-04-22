#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for loading and preprocessing data


This scripts aims to:
1- Load data locally
2- Preprocess data
3- Restructure data
4- Generate train-test sets in a reproducible manner


"""
import os
import sys
import argparse
import logging  # NOQA E402
import numpy as np  # NOQA E402
import pandas as pd  # NOQA E402
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import pickle
import time
import utils  # our custom functions

# Global variables
THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "Data"
)
DATAOUT_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "Data"
)


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def target_creator(subject, DATAIN_PATH=DATAIN_PATH):
    """
    """
    fmri_dir = os.path.join(DATAIN_PATH, f"{subject}","training_split","training_fmri")
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))

    for i in range(0,1):
        print("lh_fmri loaded...")
        print("lh_fmri.shape: ", lh_fmri.shape)
        print(f"lh_fmri[{i}].shape: ", lh_fmri[i].shape, "\n")
        rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
        print("rh_fmri loaded...")
        print("rh_fmri.shape: ", rh_fmri.shape)
        print(f"rh_fmri[{i}].shape: ", rh_fmri[i].shape, "\n")
        print("")

    return lh_fmri, rh_fmri


def training_data_creator(subject, DATAIN_PATH=DATAIN_PATH):
    """ """

    if not os.path.exists(
        f"../Data/{subject}/training_split/resized_training_images.pkl"
    ):
        images_dir = f"../Data/{subject}/training_split/training_images"
        # Create a dataloader that can load the images
        images = []
        for image in tqdm(os.listdir(images_dir)):
            image = Image.open(os.path.join(images_dir, image))
            image_array = np.array(image)

            ### Potential preprocessing here ###
            # Shape is (425, 425, 3) pr image.

            # print("size of image_array: ", image_array.shape)
            # resize
            # image_array = cv2.resize(image_array, (227, 227))
            images.append(image_array)

        # save images as pickle file
        with open(
            f"../Data/{subject}/training_split/resized_training_images.pkl", "wb"
        ) as f:
            pickle.dump(images, f)

    else:
        with open(
            f"../Data/{subject}/training_split/resized_training_images.pkl", "rb"
        ) as f:
            images = pickle.load(f)

    return images


def main():
    pass


if __name__ == "__main__":
    main()
