import os
import sys
import argparse
import logging  # NOQA E402
import numpy as np  # NOQA E402
import pandas as pd  # NOQA E402
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import pickle
import time
from sklearn.model_selection import train_test_split
import utils  # our custom functions

from generate_processed_data import target_creator, training_data_creator

# Global variables
THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "Data"
)


def report_descriptive_stats(X_data, y_data):


    print(f"Mean y_data: {np.mean(y_data)}")
    print(f"Std y_data : {np.std(y_data)}")
    print(f"Min y_data : {np.min(y_data)}")
    print(f"Max y_data : {np.max(y_data)}")
    print("")


if __name__ == "__main__":
    subject = "subj01"
    test = True

    y_data = target_creator(subject, test = test, merged = True)
    X_data = training_data_creator(subject, test = test)

    report_descriptive_stats(X_data, y_data)


