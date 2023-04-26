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
from sklearn.model_selection import train_test_split
import utils  # our custom functions
# import PCA and standard scalar from sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# Global variables
THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "Data"
)
DATAOUT_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "dataout"
)


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def target_creator(subject, DATAIN_PATH=DATAIN_PATH, test =False, merged = False, silent=False, verbose=False):
    """
    """
    fmri_dir = os.path.join(DATAIN_PATH, f"{subject}","training_split","training_fmri")
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    if test:
        lh_fmri = lh_fmri[:100]
        rh_fmri = rh_fmri[:100]

    i = 0
    
    if verbose:
        print("lh_fmri loaded...")
        print("lh_fmri.shape: ", lh_fmri.shape)
        print(f"lh_fmri[{i}].shape: ", lh_fmri[i].shape, "\n")
        print("rh_fmri loaded...")
        print("rh_fmri.shape: ", rh_fmri.shape)
        print(f"rh_fmri[{i}].shape: ", rh_fmri[i].shape, "\n")
        print("")

    if merged: 
        return merge_y_data((lh_fmri, rh_fmri))
    else:
        return lh_fmri, rh_fmri


def training_data_creator(subject, DATAIN_PATH=DATAIN_PATH, test =False, dont_use_images=False):
    """ """

    training_images_path = os.path.join(DATAIN_PATH, subject, "training_split", "resized_training_images.pkl")
    
    if not os.path.exists(
        training_images_path
    ) or test or dont_use_images:
        images_dir = os.path.join(DATAIN_PATH, subject, "training_split", "training_images")
        # Create a dataloader that can load the images
        images = [] #np.array([])
        for image in tqdm(os.listdir(images_dir)):
            if test and len(images) == 100:
                break
            if dont_use_images:
                images.append(np.array([0]))
                continue
            image = Image.open(os.path.join(images_dir, image))
            image_array = np.array(image)

            ### Potential preprocessing here ###
            # Shape is (425, 425, 3) pr image.

            # print("size of image_array: ", image_array.shape)
            # resize
            # image_array = cv2.resize(image_array, (227, 227))
            images.append(image_array)


        # save images as pickle file
        if not test or dont_use_images:
            with open(
                training_images_path, "wb"
            ) as f:
                pickle.dump(images, f)

    else:
        with open(
            training_images_path, "rb"
        ) as f:
            images = pickle.load(f)

    return images


def split_y_data(subject, y_data: np.array(object)):
    """
    Splits y_data into dictionary containing left and right splits
    
    Expected y_data format: a numpy array of all the merged fmri data 
    (e.g. as outputed by CNN model,for instance y_data shape: (N, 39548), 
    where N is the number of images
    """
    shapes = utils.get_fMRI_shapes(subject)
    len_of_left_side = shapes['left'][1] # get lh number of vertices
    
    lh_y = y_data[:, :len_of_left_side]
    rh_y = y_data[:, len_of_left_side:]
    
    return {'left': lh_y, 'right': rh_y}

def merge_y_data(y_data: tuple):
    """
    Expected format: a tuple of two numpy arrays such as: y_data = (lh_fmri, rh_fmri)
    """
    y_data_tmp = []
    for i in range(len(y_data[0])):
        y_data_tmp.append(np.concatenate((y_data[0][i], y_data[1][i]), axis=None))

    return np.array(y_data_tmp)
    

def create_train_test_split(X_data, y_data, test_size=0.20, random_state=123):
    """
    Create train-test split
    """

    if type(X_data):
        # transform to numpy array
        X_data = np.array(X_data)

    print("X_data.shape: ", X_data.shape)
    print("y_data.shape: ", y_data.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size,
                                                        random_state=random_state, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size,
                                                        random_state=random_state, shuffle=True)

    return X_train, X_val, X_test, y_train, y_val, y_test 


def StandardScaler_fit_transform(y_train, subject):
    scalar_file_path = os.path.join(DATAIN_PATH, subject, "Scalar_transformer.pickle")
    scalar_model = StandardScaler()
    y_train = scalar_model.fit_transform(y_train)
    with open(scalar_file_path, 'wb') as f:
        pickle.dump(scalar_model, f)
    return y_train

def StandardScaler_transform(y_data, subject):
    scalar_file_path = os.path.join(DATAIN_PATH, subject, "Scalar_transformer.pickle")
    with open(scalar_file_path, 'rb') as f:
        scalar_model = pickle.load(f)
    y_data = scalar_model.transform(y_data)
    return y_data

def StandardScaler_inverse_transform(y_data, subject):
    scalar_file_path = os.path.join(DATAIN_PATH, subject, "Scalar_transformer.pickle")
    with open(scalar_file_path, 'rb') as f:
        scalar_model = pickle.load(f)
    y_data = scalar_model.inverse_transform(y_data)
    return y_data


def PCA_fit_transform(y_train, subject):
    pca_file_path = os.path.join(DATAIN_PATH, subject, "PCA_model.pickle")
    pca_model = PCA(n_components=0.95)
    y_train = StandardScaler_fit_transform(y_train, subject)
    y_train = pca_model.fit_transform(y_train)
    with open(pca_file_path, 'wb') as f:
        pickle.dump(pca_model, f)
    return y_train

def PCA_transform(y_data, subject):
    pca_file_path = os.path.join(DATAIN_PATH, subject, "PCA_model.pickle")
    with open(pca_file_path, 'rb') as f:
        pca_model = pickle.load(f)
    y_data = StandardScaler_transform(y_data, subject)
    y_data = pca_model.transform(y_data)
    return y_data

def PCA_inverse_transform(y_data, subject):
    pca_file_path = os.path.join(DATAIN_PATH, subject, "PCA_model.pickle")
    with open(pca_file_path, 'rb') as f:
        pca_model = pickle.load(f)
    y_data = pca_model.inverse_transform(y_data)
    y_data = StandardScaler_inverse_transform(y_data, subject)
    return y_data


def main():
    pass


if __name__ == "__main__":
    subject = 'subj01'
    test = False
    # y_data = target_creator(subject, test = test, merged = True)
    X_data = training_data_creator(subject, test = test, dont_use_images = True)

    # input_shape = X_data[0].shape

    # X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X_data, y_data, test_size=0.2, random_state=123)
