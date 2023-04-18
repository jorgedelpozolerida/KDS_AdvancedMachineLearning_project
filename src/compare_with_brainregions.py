#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to perform all brain vision regions to fMRI response comparison 


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

    input_path = args.in_dir
    output_dir = args.out_dir

    return


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()


    parser.add_argument('--in_dir', type=str, default=None,
                        help='Path to the input directory')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Path to the output directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
