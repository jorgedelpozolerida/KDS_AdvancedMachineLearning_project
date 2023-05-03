#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script containing functions ot be used across files


Scripts assumes it is placed under src/ folder and that data is under datain/


Definitions:
- ROI: region of interest
- ROI mapping: integer that tells index of ROI within ROI class
- fsaverage space: template onto which data from all subjects is normalized

"""
# system
import os
import sys
import argparse
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # forces CPU use because errors with GPU


import logging  # NOQA E402
import numpy as np  # NOQA E402
import pandas as pd  # NOQA E402

import generate_processed_data

import re
import pickle
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from nilearn import surface, datasets, plotting
import tensorflow as tf


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


# -----------------------------
# Data loading and saving
# -----------------------------


def load_fMRIdata(subject,data_path=DATAIN_PATH):
    """
    Retrieves fMRI data for subject for both hemispheres in a dict
    """
    keys = ['left', 'right']
    fmri_data = {}
    fmri_dir = os.path.join(data_path, subject, "training_split", "training_fmri")

    for key in keys:
        fmri = np.load(os.path.join(fmri_dir, f"{key[0]}h_training_fmri.npy"))
        fmri_data[key] = fmri
    return fmri_data

def get_fMRI_shapes(subject,data_path=DATAIN_PATH):
    """
    Retrieves fMRI data shapes for subject for both hemispheres in a dict
    """
    fmri_dir = os.path.join(data_path, subject, "training_split", "training_fmri")
    keys = ['left', 'right']
    shapes = {}
    
    for key in keys:
        file_path = os.path.join(fmri_dir,f"{key[0]}h_training_fmri.npy")
        shape = np.load(file_path).shape
        shapes[key] = shape

    return shapes

def load_allvertices(subject, data_path=DATAIN_PATH):
    """
    Returns dicts with all vertices available for left and right hemisphere for given subject
    """
    keys = ["left", "right"]
    vertices = {}
    for key in keys:
        # Load all brain surface indices from all available vertices for selected subject
        vertices_path = os.path.join(
            data_path,
            subject,
            "roi_masks",
            key[0] + "h.all-vertices_fsaverage_space.npy",
        )
        vertices_data = np.load(vertices_path)
        vertices[key] = vertices_data

    return vertices


def get_ROIindices(subject, roi, data_path=DATAIN_PATH):
    """
    Retrieves indices in ROI class brain surface map (to be used to filter for some specific ROI only)
    """
    # Get ROI class based on the selected ROI
    roi_class = get_ROIclass(roi)

    # Load the ROI brain surface maps
    roiclass_mapping_path = os.path.join(
        data_path, subject, "roi_masks", f"mapping_{roi_class}.npy"
    )
    roiclass_mapping = np.load(roiclass_mapping_path, allow_pickle=True).item()

    # Select the vertices corresponding to the ROI of interest within a ROI class
    roi_indices = list(roiclass_mapping.keys())[
        list(roiclass_mapping.values()).index(roi)
    ]

    return roi_indices


def load_ROIclass_brainsurface_mask(subject, roi_class, hemisphere, space):
    """Returns brain surface indices of all ROIs belonging to a ROI class selected,
    for given subject, hemisphere and selected space

    Args:
        roi_class
        space: one of 'fsaverage' or 'challenge'
    """

    # Load the brain surface maps for selected roi class
    roiclass_mask_dir = os.path.join(
        DATAIN_PATH,
        subject,
        "roi_masks",
        f"{hemisphere[0]}h.{roi_class}_{space}_space.npy",
    )
    roiclass_mask = np.load(roiclass_mask_dir)

    return roiclass_mask


def load_ROI_brainsurface_masks(subject, roi):
    """
    Returns a dict with the brain surface masks for a given ROI, with a key
    for each of the spaces: 'fsaverage', 'challenge'. Inside another dict for data
    of each hemisphere: 'left' and 'right'
    For given subject and hemisphere
    """

    spaces = ["fsaverage", "challenge"]
    hemispheres = ["left", "right"]
    roi_brainsurfacemasks = {}

    for space in spaces:
        space_data = {}
        for hemisphere in hemispheres:
            roi_class = get_ROIclass(roi)

            roiclass_brainsurfacemap = load_ROIclass_brainsurface_mask(
                subject, roi_class, hemisphere, space
            )
            roi_indices = get_ROIindices(subject, roi, data_path=DATAIN_PATH)

            roi_brainsurfacemap = np.asarray(
                roiclass_brainsurfacemap == roi_indices, dtype=int
            )
            space_data[hemisphere] = roi_brainsurfacemap
        roi_brainsurfacemasks[space] = space_data

    return roi_brainsurfacemasks, roi_indices


def get_file_with_id(id, folder_path):
    """
    Get the file name that starts with the given ID in the given folder.

    Args:
    id (int or str): The ID to search for.
    folder_path (str): The path to the folder to search in.

    Returns:
    str: The file name that starts with the given ID, or None if no file is found.
    """
    pattern = re.compile(f"^{id}_")
    for filename in os.listdir(folder_path):
        if pattern.match(filename):
            return os.path.join(folder_path, filename)
    return None


# -----------------------------
# Visualization
# -----------------------------


def visualize_brainresponse(
    hemisphere, surface_map, title="", open_in_browser=True, cmap="cool",
    vmin = None, vmax = None 
):
    """
    Visualizes intractively the fMRI response onto brain surface in fsaverage space.
    For given hemisphere.
    """
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
    view = plotting.view_surf(
        surf_mesh=fsaverage["infl_" + hemisphere],
        surf_map=surface_map,
        bg_map=fsaverage["sulc_" + hemisphere],
        threshold=1e-14,
        cmap=cmap,
        colorbar=True,
        title=title,
    )

    if vmax is not None:
        view.set_vmax(vmax)
    if vmin is not None:
        view.set_vmin(vmin)

    if open_in_browser:
        view.open_in_browser()


# -----------------------------
# Manipulation
# -----------------------------


def get_ROIclass(roi):
    """
    Takes a region of interest (ROI) and returns to which ROI class it belongs to
    """
    if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
        roi_class = "prf-visualrois"
    elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
        roi_class = "floc-bodies"
    elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
        roi_class = "floc-faces"
    elif roi in ["OPA", "PPA", "RSC"]:
        roi_class = "floc-places"
    elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
        roi_class = "floc-words"
    elif roi in [
        "early",
        "midventral",
        "midlateral",
        "midparietal",
        "ventral",
        "lateral",
        "parietal",
    ]:
        roi_class = "streams"

    return roi_class


def map_fMRI_to_surface(subject, vertices, fmri_data, img_id, masks=None):
    """
    This functions maps fMRI data (or some mask) to brain surface in fsaverage for both hemispheres,
    for given subject
    fevfvdada
    dcdc

    """
    #
    keys = ["left", "right"]
    response_maps = {}

    # Map the fMRI data onto the brain surface map for both hemispheres
    for key in keys:
        fsaverage_response = np.zeros(len(vertices[key]))
        if masks is None:
            fsaverage_response[np.where(vertices[key])[0]] = fmri_data[key][img_id]
        else:
            fsaverage_response[np.where(vertices[key])[0]] = fmri_data[key][
                img_id, np.where(masks["challenge"][key])[0]
            ]
        response_maps[key] = fsaverage_response

    return response_maps

def ensure_dir(dir):
    ''' 
    Returns input dir and creates it if it does not exist
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Created following dir: {dir}")

    return dir 


def get_nextid(dir):
    '''
    Looks into files within dir that have a naming convention of id_XXX and
    returns next available index (integer)
    
    Ex: inside some prediction folder 1_XXX.pickle, 2_XXX.pickle it returns 3
    '''

    filenames = os.listdir(dir)
    # Get a list of IDs from the filenames
    ids = [int(filename.split('_')[0]) for filename in filenames]
    print(ids)
    # Find the next available ID
    next_id = max(ids) + 1

    return next_id


# -----------------------------
#  ML functions
# -----------------------------

def find_latest_model(model_path):
    """
    Find the latest model
    """
    for part in ["models", "predictions"]:
        tmp_model_path = ensure_dir(model_path.replace("models", part))

    model_list = os.listdir(model_path)
    if len(model_list) == 0:
        return 0

    model_list = [int(model.split('_')[1].split(".")[0]) for model in model_list]
    latest_model = max(model_list)
    return latest_model


def check_for_GPU():
    """
    Check if GPU is available
    """
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))


def predict_from_savedmodel(X_data, subject, model_name, id, save=False, recalculate=True):
    '''
    Predicts/reads prediction for subject, model and id combination.
    If save=True, it saves into file. 
    If recalculate, it will not read if file exists but rather calculate again
    '''
    
    # If file eists read it unless recalculate=True
    prediction_dir = os.path.join(DATAOUT_PATH, "predictions", model_name, subject)
    prediction_path = os.path.join(prediction_dir, f"posterior_prediction_{id}.pickle" )
    
    if os.path.exists(prediction_path) and not recalculate:
        with open(prediction_path, "rb") as f:
            # Load the already existing prediction
            y_pred = pickle.load(f)
            _logger.info(f"Loading already existing predicition file: {prediction_path}")
            return y_pred
            
    # Convert data, load model and predict        
    X_data = np.array(X_data)
    
    model_dir = os.path.join(DATAOUT_PATH, "models", model_name, subject)
    model_path = next((os.path.join(model_dir, filename) for filename in os.listdir(model_dir) if filename.endswith(f'_{id}.h5')), None)
    print(model_path)
    model = tf.keras.models.load_model(model_path)
    
    y_pred = model.predict(X_data)
   
    # Save if required
    if save:
        with open(prediction_path, "wb") as f:
            pickle.dump(y_pred, f) 
            _logger.info(f"Saved into file: {prediction_path}")  
   
    return y_pred