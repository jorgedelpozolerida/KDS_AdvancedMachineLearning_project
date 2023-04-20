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
from scipy.stats import pearsonr as corr
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


# Global variables
THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "datain"
)
DATAOUT_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "dataout"
)


device = 'cpu' #@param ['cpu', 'cuda'] {allow-input: true}
device = torch.device(device)




class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(device)
        return img



def fit_pca(feature_extractor, dataloader, batch_size):

    # Define PCA parameters
    pca = IncrementalPCA(n_components=100, batch_size=batch_size)

    # Fit PCA to batch
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Fit PCA to batch
        pca.partial_fit(ft.detach().cpu().numpy())
    return pca



def extract_features(feature_extractor, dataloader, pca):

    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        features.append(ft)
    return np.vstack(features)



def main(args):

    # input_path = args.in_dir
    # output_dir = args.out_dir
    test_img_dir = '/home/jorge/Insync/jorgitoje@gmail.com/OneDrive/Documentos/JORGE/EDUCATION/MASTER_DATASCIENCE/Semester2/AdvancedMachineLearning/MiniProject/KDS_AdvancedMachineLearning_project/datain/subj01/test_split/test_images'
    training_img_dir = '/home/jorge/Insync/jorgitoje@gmail.com/OneDrive/Documentos/JORGE/EDUCATION/MASTER_DATASCIENCE/Semester2/AdvancedMachineLearning/MiniProject/KDS_AdvancedMachineLearning_project/datain/subj01/training_split/training_images'
    data_fmri = utils.load_fMRIdata('subj01')
    print(data_fmri['left'].shape, data_fmri['right'].shape)
    
    
    rand_seed = 123 #@param
    train_img_list = os.listdir(training_img_dir)
    test_img_list  = os.listdir(test_img_dir)

    # Calculate how many stimulus images correspond to 90% of the training data
    num_train = int(np.round(len(train_img_list) / 100 * 90))
    
    # Shuffle all training stimulus images
    idxs = np.arange(len(train_img_list))
    
    print(len(train_img_list), len(test_img_list))
    np.random.shuffle(idxs)
    # Assign 90% of the shuffled stimulus images to the training partition,
    # and 10% to the test partition
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
    # No need to shuffle or split the test stimulus images
    idxs_test = np.arange(len(test_img_list))

    print('Training stimulus images: ' + format(len(idxs_train)))
    print('\nValidation stimulus images: ' + format(len(idxs_val)))
    print('\nTest stimulus images: ' + format(len(idxs_test)))




    transform = transforms.Compose([
        transforms.Resize((224,224)), # resize the images to 224x24 pixels
        transforms.ToTensor(), # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
    ])
    
    
    
    
    batch_size = 300 #@param
    # Get the paths of all image files
    train_imgs_paths = sorted(list(Path(training_img_dir).iterdir()))
    test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

    # The DataLoaders contain the ImageDataset class
    train_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_train, transform), 
        batch_size=batch_size
    )
    val_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_val, transform), 
        batch_size=batch_size
    )
    test_imgs_dataloader = DataLoader(
        ImageDataset(test_imgs_paths, idxs_test, transform), 
        batch_size=batch_size
    )
    
    
    
    lh_fmri_train = data_fmri['left'][idxs_train]
    lh_fmri_val = data_fmri['left'][idxs_val]
    rh_fmri_train = data_fmri['right'][idxs_train]
    rh_fmri_val = data_fmri['right'][idxs_val]
    
    
    del data_fmri
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
    model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
    model.eval() # set the model to evaluation mode, since
    
    
    train_nodes, _ = get_graph_node_names(model)
    print(train_nodes)

    model_layer = "features.2" #@param ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"] {allow-input: true}
    feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])

    pca = fit_pca(feature_extractor, train_imgs_dataloader)
    
    
    features_train = extract_features(feature_extractor, train_imgs_dataloader, pca)
    features_val = extract_features(feature_extractor, val_imgs_dataloader, pca)
    features_test = extract_features(feature_extractor, test_imgs_dataloader, pca)

    print('\nTraining images features:')
    print(features_train.shape)
    print('(Training stimulus images × PCA features)')

    print('\nValidation images features:')
    print(features_val.shape)
    print('(Validation stimulus images × PCA features)')

    print('\nTest images features:')
    print(features_val.shape)
    print('(Test stimulus images × PCA features)')
    
    
    del model, pca
    
    
    # Fit linear regressions on the training data
    reg_lh = LinearRegression().fit(features_train, lh_fmri_train)
    reg_rh = LinearRegression().fit(features_train, rh_fmri_train)
    # Use fitted linear regressions to predict the validation and test fMRI data
    lh_fmri_val_pred = reg_lh.predict(features_val)
    lh_fmri_test_pred = reg_lh.predict(features_test)
    rh_fmri_val_pred = reg_rh.predict(features_val)
    rh_fmri_test_pred = reg_rh.predict(features_test)

def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    # parser.add_argument('--test_img_dir', type=str, default=None,
    #                     help='Path to the folder with test images')

    # parser.add_argument('--training_img_dir', type=str, default=None,
    #                     help='Path to the fodler with training images')
    # parser.add_argument('--rand_seed', type=str, default=None,
    #                     help='Random seed used')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)