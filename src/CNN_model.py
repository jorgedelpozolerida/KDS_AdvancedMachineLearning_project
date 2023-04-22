#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for training selected model with selected parameters 


{Long Description of Script}
"""

import os
import sys
import argparse


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import utils

import tensorflow as tf
from tensorflow.keras import layers, models

from generate_processed_data import target_creator, training_data_creator
from MLdata_processing import create_train_test_split


def create_cnn_model(input_shape, output_dim):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(output_dim)
    ])

    model.summary()

    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    """
    Train the model
    """
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs,
              validation_data=(X_val, y_val), batch_size=batch_size)

    return model


def test_model(model, X_test, y_test):
    """
    Test the model
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)


if __name__ == '__main__':
    
    subject = 'subj01'
    y_data = target_creator(subject)
    X_data = training_data_creator(subject)

    input_shape = X_data[0].shape

    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X_data, y_data, test_size=0.2, random_state=123)

    # Clear RAM
    del X_data
    del y_data

    model = create_cnn_model(input_shape, output_dim)
    model = train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)

    test_model(model, X_test, y_test)

    

