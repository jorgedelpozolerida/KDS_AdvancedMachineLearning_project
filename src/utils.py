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


import logging  # NOQA E402
import numpy as np  # NOQA E402
import pandas as pd  # NOQA E402

import cv2
import pickle
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from nilearn import surface, datasets, plotting

# from decord import cpu
# from decord import VideoReader

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Global variables
THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "Data"
)
DATAOUT_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "Data"
)


# TODO: perhaps organize datain differently? A bit messy like this


# -----------------------------
# Data loading and saving
# -----------------------------


def load_fMRIdata(subject,data_path=DATAIN_PATH):
    """
    Retrieves fMRI data for subject for both hemispheres in a dict
    """
    # fmri_dir = f"../Data/{subject}/training_split/training_fmri"
    fmri_dir = os.path.join(data_path, subject, "training_split", "training_fmri")
    lh_fmri = np.load(os.path.join(fmri_dir, "lh_training_fmri.npy"))
    rh_fmri = np.load(os.path.join(fmri_dir, "rh_training_fmri.npy"))

    return {"left": lh_fmri, "right": rh_fmri}


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


# -----------------------------
# Visualization
# -----------------------------


def visualize_brainresponse(
    hemisphere, surface_map, title="", open_in_browser=True, cmap="cool"
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


# -----------------------------
#  ML functions
# -----------------------------


def find_latest_model(model_path):
    """
    Find the latest model
    """
    for part in ["models", "predictions"]:
        tmp_model_path = model_path.replace("models", part)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    model_list = os.listdir(model_path)
    if len(model_list) == 0:
        return 0

    model_list = [int(model.split('_')[1].split(".")[0]) for model in model_list]
    latest_model = max(model_list)
    return latest_model

