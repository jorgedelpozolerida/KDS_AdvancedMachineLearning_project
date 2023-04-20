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

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main(subject, roi, hemisphere):
    # Visualize a chosen ROI on a brain surface map in fsaverage space (we visualize the mask)

    roi_brainsurface_masks, roi_indices = utils.load_ROI_brainsurface_masks(
        subject, roi
    )

    view = utils.visualize_brainresponse(
        hemisphere,
        title=f"{roi}, {hemisphere} hemisphere",
        surface_map=roi_brainsurface_masks["fsaverage"][hemisphere],
    )

    # Visualize fMRI training image responses of all vertices on a brain surface map
    img_id = 0

    fmri_data = utils.load_fMRIdata(subject)

    all_vertices = utils.load_allvertices(subject)
    response_map = utils.map_fMRI_to_surface(subject, all_vertices, fmri_data, img_id)

    view = utils.visualize_brainresponse(
        hemisphere,
        surface_map=response_map[hemisphere],
        cmap="cold_hot",
        title=f"fMRI response for image {img_id}",
    )

    # Visualize the fMRI image responses of a chosen ROI on a brain surface map
    img_id = 0
    print(roi_brainsurface_masks["challenge"]["left"])
    roi_response_map = utils.map_fMRI_to_surface(
        subject,
        roi_brainsurface_masks["fsaverage"],
        fmri_data,
        img_id,
        masks=roi_brainsurface_masks,
    )

    view = utils.visualize_brainresponse(
        hemisphere,
        surface_map=roi_response_map[hemisphere],
        cmap="cold_hot",
        title=f"fMRI image response for {subject} in ROI:{roi}, {hemisphere} hemisphere,  for image {img_id}",
    )


if __name__ == "__main__":
    subject = "subj01"
    roi = "EBA"
    hemisphere = "left"
    main(subject, roi, hemisphere)
