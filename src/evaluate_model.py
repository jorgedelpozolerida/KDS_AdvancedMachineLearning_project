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
# import torch
# from torch.utils.data import DataLoader, Dataset
# from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
# from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
import pickle
from generate_processed_data import PCA_inverse_transform

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



def calculate_correlations(groundtruth_fmri, predicted_fmri, subject, save=False, save_args=None , recalculate=False):
    '''
    Returns dict for left and right correlation values computed per vertex and across images
    
    
    save_ags: {'model_name': XX, 'id': X}
    '''

    out_dir = utils.ensure_dir(os.path.join(DATAOUT_PATH, 'evaluation', save_args['model_name'], 'pearson_correlation', subject))
    file_path = os.path.join(out_dir, f"{save_args['id']}_pearsoncorr_{subject}_{save_args['model_name']}.pickle")
    
    if os.path.exists(file_path) and not recalculate:
        correlation_data = np.load(file_path, allow_pickle=True)
        _logger.info(f"Loading already calcualted correlaiton data form {file_path}")
        return correlation_data
    
    correlation_data = {}
    for hemisphere, hem_data in groundtruth_fmri.items():
        
        correlation_data_hemisphere = np.zeros(hem_data.shape[1])
        for v in tqdm(range(hem_data.shape[1]), desc="Calculating correlation"):
            correlation_data_hemisphere[v] = corr(predicted_fmri[hemisphere][:,v], hem_data[:,v])[0]
        correlation_data[hemisphere] = correlation_data_hemisphere
    
    if save:
        
        assert save_args is not None, "Please provide saving arguments accordingly"
        out_dir = utils.ensure_dir(os.path.join(DATAOUT_PATH, 'evaluation', save_args['model_name'], 'pearson_correlation', subject))
        file_path = os.path.join(out_dir, f"{save_args['id']}_pearsoncorr_{subject}_{save_args['model_name']}.pickle")
        with open(file_path, 'wb') as f:
            pickle.dump(correlation_data, f)
    
        
    return correlation_data

# TODO: noise ceiling
def calculate_noise_ceiling():
    
    return None



def square_and_normalize(correlation_list, noise_ceiling_list):
    """
    Square the correlation coefficients and normalize by the noise ceiling.
    """
    squared_correlations = np.square(correlation_list)
    normalized_correlations = squared_correlations / noise_ceiling_list
    return normalized_correlations

def load_correlations( subject, model, id):
    '''
    Loads some specific correlation data: subejct, model and ide to be specified
    '''
    out_dir = os.path.join(DATAOUT_PATH, 'evaluation', model, 'pearson_correlation', subject)
    file_path = utils.get_file_with_id(id, out_dir )
    correlation_data = np.load(file_path, allow_pickle=True)
    
    return correlation_data


def plot_ROI_correlations(subject, correlations, save=False, save_args=None, show=True):
    '''
    Function that plots obtained calculated correlations per ROI class for 
    some subject
    
    save_ags: {'model_name': XX, 'id': X}
    '''
    subject_dir = os.path.join(DATAIN_PATH, subject)
    lh_correlation, rh_correlation = correlations['left'], correlations['right']
    
    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(subject_dir, 'roi_masks', r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(subject_dir, 'roi_masks',
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(subject_dir, 'roi_masks',
            rh_challenge_roi_files[r])))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                rh_roi_correlation.append(rh_correlation[rh_roi_idx])
    roi_names.append('All vertices')
    lh_roi_correlation.append(lh_correlation)
    rh_roi_correlation.append(rh_correlation)


    lh_median_roi_correlation = [np.median(lh_roi_correlation[r]) for r in range(len(lh_roi_correlation))]
    rh_median_roi_correlation = [np.median(rh_roi_correlation[r]) for r in range(len(rh_roi_correlation))]
    
    
    # Create plot
    plt.figure(figsize=(18,6))
    x = np.arange(len(roi_names))
    width = 0.30
    plt.bar(x - width/2, lh_median_roi_correlation, width, label='Left Hemisphere')
    plt.bar(x + width/2, rh_median_roi_correlation, width,
        label='Right Hemishpere')
    plt.xlim(left=min(x)-.5, right=max(x)+.5)
    plt.xlabel('ROIs')
    plt.ylim(bottom=0, top=1)
    plt.xticks(ticks=x, labels=roi_names, rotation=60)
    plt.ylabel('Median Pearson\'s $r$')
    plt.legend(frameon=True, loc=1)
    

    if save:
        assert save_args is not None, "Please provide save_args accordingly"
        out_dir = utils.ensure_dir(os.path.join(DATAOUT_PATH, 'evaluation', save_args['model_name'], 'pearson_correlation', subject))
        plt.savefig(os.path.join(out_dir, f"{save_args['id']}_plot_pearsoncorr_{subject}_{save_args['model_name']}.png"))
   
    if show:
        plt.show()


def main(args):
    
    subject = 'subj01' # subject to get predicitons and ground truth from
    idx = 2 # id of the model run
    model = 'CNN' # model to be evaluated

    # Load predictions 
    with open (f"../dataout/predictions/{model}/{subject}/y_test_{model}_{idx}.pickle", "rb") as f:
        y_test = pickle.load(f)
    # if y_test.shape[1] == 2561:
    #     y_test = PCA_inverse_transform(y_test, subject) # PCA inverse transform
    #     # pickle the file 
    #     with open (f"../dataout/predictions/{model}/{subject}/y_test_{model}_{idx}.pickle", "wb") as f:
    #         pickle.dump(y_test, f)
    print("Shape of y_test: ", y_test.shape)

    # Load test data
    with open (f"../dataout/predictions/{model}/{subject}/y_pred_{model}_{idx}.pickle", "rb") as f:
        y_pred = pickle.load(f)
    print("Shape of y_pred: ", y_pred.shape)


    # TOCHECK: why shapes do not match!....
    # print("Ground truth: ", groundtruth_fmri['left'].shape, groundtruth_fmri['right'].shape)
    # print("Images: ", np.array(images_subject).shape)
    # min_slice = min(len(images_subject), groundtruth_fmri['left'].shape[0])
    # images_subject = images_subject[:min_slice]
    # groundtruth_fmri = {key: value[ :min_slice,:] for key,value in groundtruth_fmri.items()}


    # y_pred = utils.predict_from_savedmodel(images_subject, subject, model_name=model, id=idx, save=True, recalculate=False)
    predicted_fmri = generate_processed_data.split_y_data(subject, y_pred)
    groundtruth_fmri = generate_processed_data.split_y_data(subject, y_test)
        
    
    # 1. TEST DATA ----------------------------------------------------------
    
    # --------------------- CORRELATION METRICS -----------------------------

    correlations = calculate_correlations(groundtruth_fmri, predicted_fmri, subject,
                                        save=True, save_args={'id': idx, 'model_name': model},
                                        recalculate=False)
    
    # Plot correlation on brain surface for both hemispheres
        
    fsaverage_all_vertices = utils.load_allvertices(subject)
    for hemisphere, hem_corr in correlations.items():

        fsaverage_correlation = np.zeros(len(fsaverage_all_vertices[hemisphere]))
        fsaverage_correlation[np.where(fsaverage_all_vertices[hemisphere])[0]] = hem_corr
        utils.visualize_brainresponse(hemisphere, 
                                    surface_map=fsaverage_correlation, 
                                    cmap='cold_hot',
                                    title='Encoding accuracy, '+ hemisphere+' hemisphere'
                                    
                                    )
    # Plot correlation per Vertex
    plot_ROI_correlations(subject, correlations,
                          show=True,
                          save=True, save_args={'id': idx, 'model_name': model})



    # ---------------------  INFORMATION METRICS --------------------------



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
