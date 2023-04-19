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
import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import utils
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt

from visualize_data import visualize_brain
import time
# from nilearn import datasets
# from nilearn import plotting


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def target_creator(subject):
    """

    """
    fmri_dir = f"../Data/{subject}/training_split/training_fmri"
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


def training_data_creator(subject):
    """

    """
    
    images_dir = f"../Data/{subject}/training_split/training_images"
    # Create a dataloader that can load the images
    images = []
    for image in tqdm(os.listdir(images_dir)):
        image = Image.open(os.path.join(images_dir, image))
        images.append(np.array(image))

    return images


def convert_roi_to_roi_class(roi):
    if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
        roi_class = 'prf-visualrois'
    elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
        roi_class = 'floc-bodies'
    elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
        roi_class = 'floc-faces'
    elif roi in ["OPA", "PPA", "RSC"]:
        roi_class = 'floc-places'
    elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
        roi_class = 'floc-words'
    elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
        roi_class = 'streams'

    return roi_class


def roi_mapping_func(subject,hemisphere = "l",roi = "EBA"):
    """
    hemisphere can be one of these: ["left", "right"]

    roi can be one of these: 
    ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", 
    "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", 
    "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", 
    "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", 
    "midventral", "midlateral", "midparietal", "ventral", 
    "lateral", "parietal"]
    """

    # Define the ROI class based on the selected ROI
    roi_class = convert_roi_to_roi_class(roi)

    # Load the ROI brain surface maps
    challenge_roi_class_dir = os.path.join(f"../Data/{subject}/roi_masks",hemisphere[0]+'h.'+roi_class+'_challenge_space.npy')
    fsaverage_roi_class_dir = os.path.join(f"../Data/{subject}/roi_masks",hemisphere[0]+'h.'+roi_class+'_fsaverage_space.npy')
    roi_map_dir = os.path.join("../Data/subj01/roi_masks",'mapping_'+roi_class+'.npy')
    challenge_roi_class = np.load(challenge_roi_class_dir)
    fsaverage_roi_class = np.load(fsaverage_roi_class_dir)
    roi_map = np.load(roi_map_dir, allow_pickle=True).item()

    # Select the vertices corresponding to the ROI of interest
    roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
    challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
    fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)

    return roi_mapping, challenge_roi, fsaverage_roi


def map_brain_to_surface(subject, hemisphere,lh_fmri,rh_fmri,img_idx):
    roi_dir = os.path.join(f"../Data/{subject}/roi_masks",hemisphere[0]+'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    # Map the fMRI data onto the brain surface map
    fsaverage_response = np.zeros(len(fsaverage_all_vertices))
    if hemisphere == 'left':
        fsaverage_response[np.where(fsaverage_all_vertices)[0]] = lh_fmri[img_idx]
    elif hemisphere == 'right':
        fsaverage_response[np.where(fsaverage_all_vertices)[0]] = rh_fmri[img_idx]

    return fsaverage_response



def main(subject):

    lh_fmri, rh_fmri = target_creator(subject)
    # images = training_data_creator(subject)

    for roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", 
    "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", 
    "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", 
    "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", 
    "midventral", "midlateral", "midparietal", "ventral", 
    "lateral", "parietal"]:
            
    # roi = "midlateral"
        hemisphere = "left"

        roi_mapping, challenge_roi, fsaverage_roi = roi_mapping_func(subject = subject, 
                                                                    hemisphere = hemisphere, 
                                                                    roi = roi)
        # view = visualize_brain(roi, hemisphere, fsaverage_roi_map = fsaverage_roi)

        img_idx = 0
        fsaverage_roi_response = map_brain_to_surface(subject, hemisphere,lh_fmri,rh_fmri,img_idx)
        print("fsaverage_roi_response: ", fsaverage_roi_response.shape)

        view = visualize_brain(roi, hemisphere, 
                            title = f"{roi} part of the brain",
                            fsaverage_roi_map = fsaverage_roi_response)

        time.sleep(8)



if __name__ == '__main__':
    subject = 'subj01'
    main(subject)



