#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for training selected model with selected parameters 


{Long Description of Script}
"""

import os
import sys
import argparse

# forces CPU use because errors with GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import utils
import pickle

import tensorflow as tf

# get the number of CPUs specified in the job submission
try:
    num_cpus = int(os.environ['SLURM_CPUS_PER_TASK']) 
    tf.config.threading.set_inter_op_parallelism_threads(num_cpus)
    tf.config.threading.set_intra_op_parallelism_threads(num_cpus)
    print("CPUs set to same as HPC job.\n")
except:
    print("Not running on HPC.\n")

from keras import layers, models
import keras
from utils import find_latest_model
from generate_processed_data import target_creator, training_data_creator, create_train_test_split
from generate_processed_data import StandardScaler_fit_transform , StandardScaler_transform , StandardScaler_inverse_transform
from generate_processed_data import PCA_fit_transform , PCA_transform , PCA_inverse_transform


from utils import check_for_GPU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.applications import EfficientNetB0, EfficientNetB5


def create_effecientnet_model(input_shape, output_dim, model_path):
    # base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    # freeze the weights of the base model
    base_model = EfficientNetB5(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    output_layer = layers.Dense(output_dim[0])(x)
    model = models.Model(inputs=base_model.inputs, outputs=output_layer)
    model.summary()

    # Save the model
    try: 
        # If on HPC, name it the job id.
        model_version = os.environ.get('SLURM_JOB_ID')
    except:
        # If running local, call it model num.
        model_version = find_latest_model(model_path) +1

    return model, model_version


def train_model(model, model_path, X_train, y_train, X_val, model_version, y_val, epochs, batch_size,
                learning_rate=0.001, patience = 5):
    """
    Train the model
    """

    model.compile(optimizer= Adam(learning_rate = learning_rate),
                  loss= keras.losses.MeanSquaredError(),
                  metrics=['MSE','MAE','MAPE'])

    es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience , restore_best_weights = True)
    cb_list = [es_callback]

    # Train the model
    model_history = model.fit(X_train, y_train, 
                                epochs=epochs, batch_size=batch_size, 
                                validation_data=(X_val, y_val), callbacks=cb_list)

    epochs_trained = len(model_history.history['loss']) - patience
    print(f"Epochs Trained: {epochs_trained}")

    val_loss, val_mae, val_mse, val_mape = model.evaluate(X_val,  y_val,  verbose=1)

    model.save(f"{model_path}/effecientnet_{model_version}.h5")

    return model


def save_test_pred(model_path, model, X_test, y_test, model_version):
    """
    Save the test predictions
    """
    y_pred = model.predict(X_test)
    y_pred = PCA_inverse_transform(y_pred, subject) # Inverts into "NeUrOa DaTA"
    y_test = PCA_inverse_transform(y_test, subject) # Inverts into "NeUrOa DaTA"

    model_path = model_path.replace("models", "predictions")

    with open(f"{model_path}/y_test_CNN_{model_version}.pickle", "wb") as f:
        pickle.dump(y_test, f)   
    with open(f"{model_path}/y_pred_CNN_{model_version}.pickle", "wb") as f:
        pickle.dump(y_pred, f)     


def test_model(model, X_test, y_test):
    """
    Test the model
    """
    # Evaluate the model
    test_loss, test_mae, test_mse, test_mape = model.evaluate(X_test, y_test, verbose=1)

    y_pred = model.predict(X_test)
    y_pred = PCA_inverse_transform(y_pred, subject) # Inverts into "NeUrOa DaTA"
    y_test = PCA_inverse_transform(y_test, subject) # Inverts into "NeUrOa DaTA"

    model_path = model_path.replace("models", "predictions")

    with open(f"{model_path}/y_test_CNN_{model_version}.pickle", "wb") as f:
        pickle.dump(y_test, f)   
    with open(f"{model_path}/y_pred_CNN_{model_version}.pickle", "wb") as f:
        pickle.dump(y_pred, f)   
    
    return y_pred
    



if __name__ == '__main__':
    
    subject = 'subj01'
    test = False 
    y_data = target_creator(subject, test = test, merged = True)
    X_data = training_data_creator(subject, test = test)
    epochs = 20
    batch_size = 32
    learning_rate = 0.0000005
    patience = 3
    model_path = f"../dataout/models/EffecientNet/{subject}"
 
    print("############################### \n")
    print(" MODEL PARAMETERS: ")
    print("")
    print("Subject: ", subject)
    print("Test: ", test)
    print("Epochs: ", epochs)
    print("Batch Size: ", batch_size)
    print("Learning Rate: ", learning_rate)
    print("Patience: ", patience)
    print("Model Path: ", model_path)
    print("############################### \n")

    input_shape = X_data[0].shape
    output_dim = y_data[0].shape
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X_data, y_data, test_size=0.2, random_state=123)

    # Clear RAM
    del X_data
    del y_data

    # Scale and PCA transform the data 
    y_train =  PCA_fit_transform(y_train, subject)
    y_test = PCA_transform(y_test, subject)
    y_val = PCA_transform(y_val, subject)
    input_shape = X_train[0].shape
    output_dim = y_train[0].shape

    print("     Data shape : ")
    print("   Input Shape: ", input_shape)
    print("   Output Dim: ", output_dim)
    print("\n")

    check_for_GPU()
    model, model_version = create_effecientnet_model(input_shape = input_shape, 
                                                        output_dim = output_dim, 
                                                        model_path = model_path)

    model = train_model(model = model, 
                        model_path = model_path, 
                        X_train = X_train, 
                        y_train = y_train, 
                        X_val = X_val, 
                        y_val = y_val, 
                        model_version = model_version,
                        epochs = epochs, 
                        batch_size = batch_size,
                        learning_rate = learning_rate,
                        patience=patience)

    save_test_pred(model_path, model, X_test, y_test, model_version)

    

