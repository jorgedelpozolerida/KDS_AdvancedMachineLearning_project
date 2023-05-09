#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for training linearized encoding model with input parameters


{Long Description of Script}
"""
import os
import sys
import argparse


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402

import utils

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
import joblib
import pickle

from scipy.stats import pearsonr as corr
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


# Global variables
THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "Data"
)
DATAOUT_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "dataout"
)


class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform, device):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(self.device)
        return img



def extract_features_and_fit_incrementalPCA(feature_extractor, dataloader, n_components, batch_size, subject, id,  save=True, recalculate=False):
    '''
    Fits an incremental PCA to extracted features or loads one if already calculated.
    
    Behaviour can be changed based on argument values
    '''

    out_dir = utils.ensure_dir(os.path.join(DATAOUT_PATH, 'models', 'linearizing_model', subject))
    pca_path = os.path.join(out_dir, f"{id}_PCAmodel_{subject}.joblib")
    
    
    if os.path.exists(pca_path) and not recalculate:
        pca = joblib.load(pca_path)
        _logger.info(f"Loading already calculated PCA modelfrom: {pca_path}")
        return pca
    
    _logger.info("Calculating PCA")
    
    # Define PCA parameters
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # Fit PCA to batch
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Fit PCA to batch
        feature_vector = ft.detach().cpu().numpy()
        # print(feature_vector.shape)
        pca.partial_fit(feature_vector)
        
    if save:
        joblib.dump(pca, pca_path)


    return pca



def extract_features_and_project_withPCA(feature_extractor, dataloader, pca):
    '''
    Extracts features and porjects data with PCA already fitted
    '''

    features = []
    for i, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        features.append(ft)
        
    return np.vstack(features)



def perform_LinearRegression(X, y, subject, id, save=True, recalculate=False):
    '''
    Performs Linear Regression or loads one if already calculated. Important:
    calculation done for each hemisphere separately
    
    Behaviour can be changed based on argument values
    '''
    
    out_dir = utils.ensure_dir(os.path.join(DATAOUT_PATH, 'models', 'linearizing_model', subject))
    pca_path = os.path.join(out_dir, f"{id}_LinearRegression_{subject}.joblib")
    
    
    if os.path.exists(pca_path) and not recalculate:
        LR_model = joblib.load(pca_path)
        _logger.info(f"Loading already calcualted correlaiton data form {pca_path}")
        return LR_model
    
    LR_model  = LinearRegression().fit(X, y)
    
    return LR_model


def split_data(subject, device, batch_size, train_percentage=90 ):
    
    # Load data
    data_fmri = utils.load_fMRIdata(subject)

    training_img_dir = os.path.join(DATAIN_PATH,subject,'training_split/training_images')
    test_img_dir = os.path.join(DATAIN_PATH, subject,'test_split/test_images')

    
    train_img_list = os.listdir(training_img_dir)
    test_img_list  = os.listdir(test_img_dir)
    
    # Calculate how many stimulus images correspond to 90% of the training data
    num_train = int(np.round(len(train_img_list) / 100 * train_percentage))
    
    # Shuffle all training stimulus images
    idxs = np.arange(len(train_img_list))
    
    np.random.shuffle(idxs)
    
    # Assign 90% of the shuffled stimulus images to the training partition,

    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]

    print('Training stimulus images: ' + format(len(idxs_train)))
    print('\nValidation stimulus images: ' + format(len(idxs_val)))


    # Build Dataloaders
    
    transform = transforms.Compose([
        transforms.Resize((224,224)), # resize the images to 224x24 pixels
        transforms.ToTensor(), # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
    ])
    
    
    # Get the paths of all image files
    train_imgs_paths = sorted(list(Path(training_img_dir).iterdir()))

    # The DataLoaders contain the ImageDataset class
    train_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_train, transform, device), 
        batch_size=batch_size
    )
    val_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_val, transform, device), 
        batch_size=batch_size
    )
    
    
    return train_imgs_dataloader, val_imgs_dataloader, idxs_train, idxs_val
    
    
    


def main(args):

    idx = 2
    subject = 'subj01'
    
    
    
    batch_size = args.batchsize #@param # needed calculation  to avoid error from skelearn package
    n_components = args.n_components
    rand_seed = 5 #@param
    np.random.seed(rand_seed)
    device = 'cpu'
    device = torch.device(device)
    

    # Load and split data
    data_fmri = utils.load_fMRIdata('subj01')

    train_imgs_dataloader, val_imgs_dataloader, idxs_train, idxs_val = split_data(subject, device, batch_size, train_percentage=90)

    lh_fmri_train = data_fmri['left'][idxs_train]
    lh_fmri_val = data_fmri['left'][idxs_val]
    rh_fmri_train = data_fmri['right'][idxs_train]
    rh_fmri_val = data_fmri['right'][idxs_val]
    
    
    
    # PART 1 - Extract and downsample image features from AlexNet --------------
    
    
    # Load AlexNet and create feature extractor
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
    model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
    model.eval() # set the model to evaluation mode, since not training it



    model_layer = args.alexnet_layer
    feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])

    # Fit incremental PCA to features
    pca = extract_features_and_fit_incrementalPCA(feature_extractor, train_imgs_dataloader,
                                                  n_components, batch_size,  subject, id=idx, 
                      save=True, 
                      recalculate=False)
    
    # Project features to new space
    features_train = extract_features_and_project_withPCA(feature_extractor, train_imgs_dataloader, pca)
    features_val = extract_features_and_project_withPCA(feature_extractor, val_imgs_dataloader, pca)

    print('\nTraining images features:')
    print(features_train.shape)
    print('(Training stimulus images × PCA features)')

    print('\nValidation images features:')
    print(features_val.shape)
    print('(Validation stimulus images × PCA features)')

    print('\nTest images features:')
    print(features_val.shape)
    print('(Test stimulus images × PCA features)')
  
    
    
    # PART 2 - Linearly map the AlexNet image features to fMRI responses -------
     

    # Fit linear regressions on the training data for both hemispheres
    reg = LinearRegression().fit(features_train, np.concatenate((lh_fmri_train, rh_fmri_train), axis=1))
    # Use fitted linear regressions to predict the validation and test fMRI data
    y_pred = reg.predict(features_val)
    
    y_val = np.concatenate((lh_fmri_val, rh_fmri_val), axis=1)
    
    # Save the variables to pickle files
    dir_out = utils.ensure_dir(os.path.join(DATAOUT_PATH, "predictions", "linearizing_model", subject))
    
    with open(os.path.join(dir_out, f"y_pred_linearizing_model_{idx}.pickle"), "wb") as f:
        pickle.dump(y_pred, f)

    with open(os.path.join(dir_out, f"y_test_linearizing_model_{idx}.pickle"), "wb") as f:
        pickle.dump(y_val, f)



def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_components', type=int, default=98,
                        help='P')

    parser.add_argument('--batchsize', type=int, default=300,
                        help='')

    parser.add_argument('--alexnet_layer', type=str, default="features.2",
                        help='')
    # parser.add_argument('--rand_seed', type=str, default=None,
    #                     help='Random seed used')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)