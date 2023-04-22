#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for training selected model with selected parameters 


{Long Description of Script}
"""

import os
import sys
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # forces CPU use because errors with GPU


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import utils

import tensorflow as tf
from keras import layers, models
import keras
from utils import find_latest_model

from generate_processed_data import target_creator, training_data_creator, create_train_test_split
from keras.optimizers import Adam


def check_for_GPU():
    """
    Check if GPU is available
    """
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))


def create_cnn_model(input_shape, output_dim):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(4, (5, 5), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(8, (5, 5), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(output_dim[0])
    ])

    model.summary()

    return model


def train_model(model, model_path, X_train, y_train, X_val, y_val, epochs, batch_size,
                learning_rate=0.001, patience = 5):
    """
    Train the model
    """

    model.compile(optimizer= Adam(learning_rate = learning_rate),
                  loss= keras.losses.MeanSquaredError(),
                  metrics=['MSE','MAE'])

    es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience , restore_best_weights = True)
    cb_list = [es_callback]

    # Train the model
    model_history = model.fit(X_train, y_train, 
                                epochs=epochs, batch_size=batch_size, 
                                validation_data=(X_val, y_val), callbacks=cb_list)

    

    # print("model.keys():", model.History.keys())
    epochs_trained = len(model_history.history['loss']) - patience
    val_loss, val_mae, val_mse  = model.evaluate(X_val,  y_val,  verbose=1)
    # Save the model


    model.save(f"{model_path}/model_{find_latest_model()}.h5")

    return model


def save_test_pred_(model_path, specific_model_path, specific_model_name):
    """
    Save the test predictions
    """
    y_pred = test_model(model, X_test, y_test)
    


def test_model(model, X_test, y_test):
    """
    Test the model
    """

    # Evaluate the model
    test_loss,test_mae,test_mse = model.evaluate(X_test, y_test, verbose=1)
    y_pred = model.predict(X_test)
    
    return y_pred
    


if __name__ == '__main__':
    
    subject = 'subj01'
    test = False
    y_data = target_creator(subject, test = test, merged = True)
    X_data = training_data_creator(subject, test = test)
    epochs = 500
    batch_size = 16
    model_path = f"../OutData/{subject}/CNN"

    input_shape = X_data[0].shape
    output_dim = y_data[0].shape
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X_data, y_data, test_size=0.2, random_state=123)

    # Clear RAM
    del X_data
    del y_data

    check_for_GPU()
    model = create_cnn_model(input_shape, output_dim)
    model = train_model(model, model_path, X_train, y_train, X_val, y_val, epochs, batch_size,
                        learning_rate = 0.0001)

    test_model(model, X_test, y_test, model_path)

    

