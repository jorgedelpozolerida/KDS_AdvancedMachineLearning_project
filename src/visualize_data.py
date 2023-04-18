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


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def main(args):

    # input_path = args.in_dir
    # output_dir = args.out_dir
    # Select ROI
    ROI = 'WB'  #@param ["WB", "V1", "V2","V3", "V4", "LOC", "EBA", "FFA","STS", "PPA"]
    fmri_train_all, _ = utils.get_fmri(
        '/home/jorge/Insync/jorgitoje@gmail.com/OneDrive/Documentos/JORGE/EDUCATION/MASTER_DATASCIENCE/Semester2/AdvancedMachineLearning/MiniProject/KDS_AdvancedMachineLearning_project/datain/participants_data_v2021/full_track/sub04',
        ROI
    )
    print(fmri_train_all.shape)
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