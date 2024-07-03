import numpy as np
import os
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from math import ceil, floor
import warnings
import matplotlib.pyplot as plt


class DataReader(object):

    def __init__(self, parent_dir=r'RawData'):
        while parent_dir.startswith('\\'):
            parent_dir = parent_dir[1:]

        if not os.path.isdir(parent_dir):
            parent_dir = os.path.join('..', parent_dir)
            if not os.path.isdir(parent_dir):
                parent_dir = os.path.join('..', parent_dir)
                if not os.path.isdir(parent_dir):
                    raise ValueError(f'No directory found: {parent_dir}')

        self.parent_dir = parent_dir

    def data_dir(self, data_group):
        return os.path.join(self.parent_dir, data_group)

    def read_raw_data(self, data_group, rep_rate, file_num=0):
        """Reads the raw data file for given frequency. No processing. Sampling rate is 20ns. """
        data_dir = self.data_dir(data_group)
        data_files = os.listdir(data_dir)

        freq_name = rf'{rep_rate}kHz'

        correct_files = []
        for file in data_files:
            if freq_name in file:
                correct_files.append(file)
        if len(correct_files) == 0:
            raise FileNotFoundError(f'No data file for {freq_name} found in {data_dir}')

        file_name = correct_files[file_num]
        data_raw = np.loadtxt(data_dir + rf'\{file_name}', delimiter=',', unpack=True)
        data_raw = data_raw.T

        return data_raw
