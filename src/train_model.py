#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for training selected model with selected parameters 


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



    return


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Dataset to train NN on, one in ["MNIST", "CIFAR10", "ImageNet"]')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Number of epochs to train the model')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Path where to save training data')
    # and so on and so forth
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)