import os
import sys
import argparse
import glob

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
import utils

# Global variables
THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "Data"
)
DATAOUT_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "dataout"
)



if __name__ == "__main__":
    subject = "subj01"
    fmri_data = utils.load_fMRIdata(subject)

    all_fmri = np.concatenate((fmri_data["left"], fmri_data["right"]), axis=1)


    # print statistics 
    print(" ### Vertex Statistics of Subject 1 ###\n\n")
    print(" Mean Vertex Value: ", np.mean(all_fmri))
    print(" Standard Deviation Vertex Value: ", np.std(all_fmri))
    print("")
    print(" Min Vertex Value: ", np.min(all_fmri))
    print(" 5th Percentile Value: ", np.quantile(all_fmri, 0.05))
    print(" 1Q Vertex Value: ", np.quantile(all_fmri, 0.25))
    print(" 2Q Vertex Value: ", np.quantile(all_fmri, 0.5))
    print(" 3Q Vertex Value: ", np.quantile(all_fmri, 0.75))
    print(" 95th Percentile Value: ", np.quantile(all_fmri, 0.95))
    print(" Max Vertex Value: ", np.max(all_fmri))
    print("")


        
                

