#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to perform all visualization of data at once and save images


{Long Description of Script}
"""

import os
import sys
import argparse


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import utils
from nilearn import datasets
from nilearn import plotting


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def visualize_brain(roi, hemisphere, fsaverage_roi_map, title = "Poop", open_in_browser=True):
    # Map the fMRI data onto the brain surface map
    # Create the interactive brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    view = plotting.view_surf(
        surf_mesh=fsaverage['infl_'+hemisphere],
        surf_map=fsaverage_roi_map,
        bg_map=fsaverage['sulc_'+hemisphere],
        threshold=1e-14,
        cmap='cold_hot',
        colorbar=True,
        title= title#roi+', '+hemisphere+' hemisphere'
        )

    if open_in_browser:
        view.open_in_browser()






if __name__ == '__main__':
    pass