import numpy as np
import os
import logging
import sys
import warnings

class DataParser(object):

    def __init__(self, sub_dir= 'raw_5', parent_dir= r'RawData'):

        while sub_dir.startswith('\\'):
            sub_dir = sub_dir[1:]
        self._sub_dir = sub_dir

        while parent_dir.startswith('\\'):
            parent_dir = parent_dir[1:]
        self._parent_dir = parent_dir

    @property
    def sub_dir(self):
        return self._sub_dir

    @sub_dir.setter
    def sub_dir(self, new_sub_dir):
        while new_sub_dir.startswith('\\'):
            new_sub_dir = new_sub_dir[1:]
        self._sub_dir = new_sub_dir

    @property
    def parent_dir(self):
        return self._parent_dir

    @property
    def data_dir(self):
        return self.parent_dir + '\\' + self.sub_dir

    def read_raw_data(self, frequency, freq_name=None, file_num=0):
        data_dir = self.data_dir

        try:
            data_files = os.listdir(data_dir)
        except FileNotFoundError:
            try:
                data_dir = '..\\' + data_dir
                data_files = os.listdir(data_dir)
            except FileNotFoundError:
                try:
                    data_dir = '..\\..\\' + data_dir
                    data_files = os.listdir(data_dir)
                except FileNotFoundError:
                    raise

        if freq_name is None:
            freq_name = rf'{frequency}kHz'

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

