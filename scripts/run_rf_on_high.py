import numpy as np
import matplotlib.pyplot as plt
import time
import os
import logging

from tes_resolver.classifier.inner_product import InnerProductClassifier
from tes_resolver.classifier.tabular import TabularClassifier
from tes_resolver.traces import Traces
import tes_resolver.config as config

from src.utils import LogUtils, DFUtils
from src.data_reader import DataReader

'''
script to produce  PN distributions using tabular classifiers, specify power, whether FE and modeltype.
'''

#TODO: if i take data more carefully this time, can I remove the need for subtract offset? Because there is no good way
# to subtract offset in actual experiment, where the data's distribution might be different from calibration, and also
# no way of subtracting offset of 0-photon traces because they are difficult to identify (or even might not exist!)

# Todo: also consider pooling together data from mulitple groups and train a classifier based on the pooled-together data.

'''Specify parameters'''
# Data
dataReader = DataReader('RawData')
datagroup = 'raw_8'
rep_rate = 700

# Tabular classifier
modeltype = 'RF'
test_size=0.1
n_estimators = 600

# Inner product classifier
multiplier=1.
num_bins = 1000

# Correct for a voltage shift between training data and actual data. Only makes sense if training data and actual data
# are from the same datagroup.
vertical_shift = True

# logging
time_stamp = config.time_stamp
results_dir = os.path.join(config.home_dir, '..', 'Results', modeltype, f'{datagroup}_{time_stamp}')

LogUtils.log_config(time_stamp='', dir=results_dir, filehead='log', module_name='', level=logging.INFO)
logging.info(rf'Produce PN distributions using tabular classifiers. Raw data from {datagroup}, ' 
             f'feature extraction is {feature_extraction}, model type is {modeltype}. \n'
             f'Test_size = {test_size}.\n'
             f'Inner product method used to identify calibration data at 100kHz with multiplier={multiplier} and num_bins={num_bins}. \n'
             f'The 100kHz calibration data is used to generate training data by overlapping them to higher frequencies. \n'
             f'pca_components={pca_components}- pca is used to clean up the 100kHz data to generate training data. '
             f'vertical_shift={vertical_shift}- this is whether the training data is shifted to meet the same average-trace height as the actual data.\n'
             f'Trigger={triggered}- this is whether theactual data are  triggered on the rising edge of traces (only for more than 300kHz). \n'
             f'Training data is shifted horizontally to match peaks with actual data'
             )

