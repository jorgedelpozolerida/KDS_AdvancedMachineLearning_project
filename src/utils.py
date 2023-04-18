#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script containing functions ot be used across files


Scripts assumed it is placed under src/ folder and that data is under /datain 

"""
# system
import os
import sys
import argparse
import glob


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402

import cv2
import pickle
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from nilearn import surface, datasets, plotting
from decord import cpu
from decord import VideoReader

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Global variables
THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join( os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), 'datain')
DATAOUT_PATH = os.path.join( os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), 'dataout')


# -----------------------------
# Data loading and saving
# -----------------------------

def save_dict(di_, filename_):
  with open(filename_, 'wb') as f:
    pickle.dump(di_, f)


def load_dict(filename_):
  with open(filename_, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    ret_di = u.load()
  return ret_di


def get_fmri(fmri_dir, ROI):
  """This function loads fMRI data into a numpy array for to a given ROI.
  Parameters
  ----------
  fmri_dir : str
    path to fMRI data.
  ROI : str
    name of ROI.
  Returns
  -------
  np.array
    matrix of dimensions #train_vids x #repetitions x #voxels
    containing fMRI responses to train videos of a given ROI
  """
  # Loading ROI data
  ROI_file = os.path.join(fmri_dir, ROI + ".pkl")
  ROI_data = load_dict(ROI_file)
  # averaging ROI data across repetitions
  ROI_data_train = np.mean(ROI_data["train"], axis=1)
  if ROI == "WB":
    voxel_mask = ROI_data['voxel_mask']
    return ROI_data_train, voxel_mask

  return ROI_data_train


def saveasnii(brain_mask, nii_save_path, nii_data):
  img = nib.load(brain_mask)
  nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
  nib.save(nii_img, nii_save_path)



# -----------------------------
# Visualization
# -----------------------------


def visualize_activity(vid_id, sub, track = "full_track"):
    
  # Setting up the paths for whole brain data
  fmri_dir = os.path.join(DATAIN_PATH, 'participants_data_v2021') # TODO: change to relative
  
  # get the right track directory depending on whole brain/ROI choice
  track_dir = os.path.join(fmri_dir, track)

  # get the selected subject's directory
  sub_fmri_dir = os.path.join(track_dir, sub)

  # result directory to store nifti file
  results_dir = DATAOUT_PATH # TODO: change to relative

  # mapping the data to voxels and storing in a nifti file
  fmri_train_all,voxel_mask = get_fmri(sub_fmri_dir, "WB")
  visual_mask_3D = np.zeros((78, 93, 71))
  visual_mask_3D[voxel_mask==1]= fmri_train_all[vid_id, :]
  brain_mask = './example.nii'
  nii_save_path = os.path.join(results_dir, 'vid_activity.nii')
  saveasnii(brain_mask, nii_save_path, visual_mask_3D)

  # visualizing saved nifti file
  plotting.plot_glass_brain(nii_save_path,
                          title='fMRI response',plot_abs=False,
                          display_mode='lyr',colorbar=True)

