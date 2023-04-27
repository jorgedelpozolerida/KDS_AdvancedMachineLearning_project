#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to perform all visualization of data at once and save images


{Long Description of Script}
"""

import os
import sys
import argparse


import logging  # NOQA E402
import numpy as np  # NOQA E402
import pandas as pd  # NOQA E402
import utils
from nilearn import datasets
from nilearn import plotting
import pickle
import generate_processed_data

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


def main():
    
    roi = "EBA"
    subject = "subj01"
    hemisphere = "left"
    model = 'CNN'
    idx = 3
    img_id_list = [0]
    
    # USING ALREADY SAVED FILES
    # Load pred data
    pred_file_path = os.path.join(DATAOUT_PATH, f"predictions/{model}/{subject}/y_pred_{model}_{idx}.pickle")
    with open (pred_file_path, "rb") as f:
        y_pred = pickle.load(f)
    print("Shape of y_pred: ", y_pred.shape)
    
    # Load test data
    gt_file_path = os.path.join(DATAOUT_PATH, f"predictions/{model}/{subject}/y_test_{model}_{idx}.pickle")
    with open (gt_file_path, "rb") as f:
        y_test = pickle.load(f)
    print("Shape of y_test: ", y_test.shape)

    # Create left-right split into dictionary
    predicted_fmri = generate_processed_data.split_y_data(subject, y_pred)
    groundtruth_fmri = generate_processed_data.split_y_data(subject, y_test)
    
    # fmri_data = utils.load_fMRIdata(subject) # ground truth
    # fmri_data = groundtruth_fmri # use GT data
    
    
    both_fmri_data = {"Predicted": predicted_fmri, "Ground truth": groundtruth_fmri} 
    
    
    for img_id in img_id_list:
        for type_data, fMRI_data in both_fmri_data.items():
            
            # Visualize a chosen ROI on a brain surface map in fsaverage space (we visualize the mask)

            roi_brainsurface_masks, roi_indices = utils.load_ROI_brainsurface_masks(
                subject, roi
            )

            # view = utils.visualize_brainresponse(
            #     hemisphere,
            #     title=f"{roi}, {hemisphere} hemisphere",
            #     surface_map=roi_brainsurface_masks["fsaverage"][hemisphere],
            # )



            # Visualize fMRI image responses of all vertices on a brain surface map
            
            all_vertices = utils.load_allvertices(subject)
            response_map = utils.map_fMRI_to_surface(subject, all_vertices, fMRI_data, img_id)

            view = utils.visualize_brainresponse(
                hemisphere,
                surface_map=response_map[hemisphere],
                cmap="cold_hot",
                title=f"fMRI response for image {img_id}. Type: {type_data}",
            )


            # ---------------------------------------
        
        
            # Visualize the fMRI image responses of a chosen ROI on a brain surface map

            print(roi_brainsurface_masks["challenge"]["left"])
            roi_response_map = utils.map_fMRI_to_surface(
                subject,
                roi_brainsurface_masks["fsaverage"],
                fMRI_data,
                img_id,
                masks=roi_brainsurface_masks,
            )

            view = utils.visualize_brainresponse(
                hemisphere,
                surface_map=roi_response_map[hemisphere],
                cmap="cold_hot",
                title=f"fMRI image response for {subject} in ROI:{roi}, {hemisphere} hemisphere, image={img_id}. Type: {type_data}",
            )


if __name__ == "__main__":

    main()
