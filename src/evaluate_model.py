#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script that evaluates model predictions against ground truth

This script aims to calculate 3 different types of metrics:

1- Correlation-based metrics:
    a. Pearson correlation coefficient: This measures the linear relationship
    b. Spearman rank correlation coefficient: This measures the monotonic relationship.

2. Information-theoretic metrics: 
    a. Mutual information: This measures the amount of information shared 
    b. Normalized mutual information: This is a normalized version of mutual information

3.  Mean squared error (MSE) measures:
Generic metric that measures the overall difference between two vectors 
and does not take into account the specific properties of the neural activity patterns being studied.
However, it can still be a useful metric for evaluating encoding models, 
especially during model training, as it provides a quantitative measure of the 
model's ability to fit the data.
"""

import os
import sys
import argparse


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402

import utils 
import generate_processed_data
from tqdm import tqdm 

import time
import datetime
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
import pickle

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



def calculate_correlation(groundtruth_fmri, predicted_fmri, subject, model, save=False, plot = False):
    '''
    Returns dict for left and right correlation values computed per vertex and across images
    '''
    # groundtruth_fmri = utils.load_fMRIdata(subject)
    # predicted_fmri = utils.load_predicted_data(subject, model='CNN')
    
    correlation_data = {}
    for hemisphere, hem_data in groundtruth_fmri.items():
        
        correlation_data_hemisphere = np.zeros(hem_data.shape[1])
        for v in tqdm(range(hem_data.shape[1])):
            correlation_data_hemisphere[v] = corr(predicted_fmri[hemisphere][:,v], hem_data[:,v])[0]
        correlation_data[hemisphere] = correlation_data_hemisphere
    
    if save:
        out_dir = utils.ensure_dir(os.path.join(DATAOUT_PATH, 'evaluation', model, 'pearson_correlation', subject))
        file_path = os.path.join(out_dir, f'pearsoncorr_{subject}_{model}_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pickle')
        with open(file_path, 'wb') as f:
            pickle.dump(correlation_data, f)
    
    if plot:   
        
        fsaverage_all_vertices = utils.load_allvertices(subject)
       
        for hemisphere, hem_corr in correlation_data.items():

            fsaverage_correlation = np.zeros(len(fsaverage_all_vertices[hemisphere]))
            fsaverage_correlation[np.where(fsaverage_all_vertices[hemisphere])[0]] = correlation_data[hemisphere]
            utils.visualize_brainresponse(hemisphere, 
                                        surface_map=fsaverage_correlation, 
                                        cmap='cold_hot',
                                        title='Encoding accuracy, '+ hemisphere+' hemisphere'
                                        )
            

    return correlation_data

def calculate_correlation2(groundtruth_fmri, predicted_fmri, subject, model, save=False, plot = False):
    '''
    Returns dict for left and right correlation values computed per vertex and across images
    '''
    # groundtruth_fmri = utils.load_fMRIdata(subject)
    # predicted_fmri = utils.load_predicted_data(subject, model='CNN')
    
    correlation_data = {}
    

    for hemisphere, hem_data in groundtruth_fmri.items():
        # Calculate correlations for each column using list comprehension and vectorized operations
        correlation_data_hemisphere = np.array([corr(predicted_fmri[hemisphere][:, v], hem_data[:, v])[0] for v in range(hem_data.shape[1])])
        correlation_data[hemisphere] = correlation_data_hemisphere

    return correlation_data

def main(args):

    subject = 'subj01'

    
    groundtruth_fmri = utils.load_fMRIdata(subject)
    predicted_fmri = utils.load_predicted_data(subject, model_name='CNN', id=1)
    

    correlation = calculate_correlation(groundtruth_fmri, predicted_fmri, subject, model='CNN', save=False, plot = False)


    return


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()


    # parser.add_argument('--in_dir', type=str, default=None,
    #                     help='Path to the input directory')
    # parser.add_argument('--out_dir', type=str, default=None,
    #                     help='Path to the output directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)