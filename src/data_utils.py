import numpy as np
import os
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from math import ceil, floor
import warnings
import matplotlib.pyplot as plt

class DataReader(object):

    def __init__(self, sub_dir= None, parent_dir= r'RawData'):

        while sub_dir.startswith('\\'):
            sub_dir = sub_dir[1:]
        self._sub_dir = sub_dir

        while parent_dir.startswith('\\'):
            parent_dir = parent_dir[1:]
        self._parent_dir = parent_dir

    @property
    def sub_dir(self):
        return self._sub_dir

    @property
    def parent_dir(self):
        return self._parent_dir

    @property
    def data_dir(self):
        return self.parent_dir + '\\' + self.sub_dir

    def read_raw_data(self, frequency, file_num=0):
        """Reads the raw data file for given frequency. No processing. Sampling rate is 20ns. """
        data_dir = self.data_dir

        try:
            data_files = os.listdir(data_dir)
        except FileNotFoundError:
            try:
                data_dir = '..\\' + data_dir
                data_files = os.listdir(data_dir)
            except FileNotFoundError:
                try:
                    data_dir = '..\\' + data_dir
                    data_files = os.listdir(data_dir)
                except FileNotFoundError:
                    raise

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

    # def parse_data(self, frequency, interpolated=False, triggered=True, **kwargs):
    #     """
    #     Return data array, where each row is a trace.
    #
    #     :param frequency: Rep rate, unit: kHz
    #     :param interpolated: If interpolated, then 500 samples per trace. Otherwise, int(5e4/frequency) samples per trace.
    #     :param triggered: The first trace is triggered on the rising edge.
    #     :param kwargs: Extra arguments in read_raw_data.
    #     """
    #
    #     data_raw = self.read_raw_data(frequency, **kwargs)
    #
    #     # interpolate the data, such that each trace has 500 sample points
    #     f = frequency // 100
    #     ex_samples, data_intp = DataChopper.interpolate_data(data_raw, f)
    #
    #     if triggered:
    #         if frequency < 300:
    #             # No triggering for lower frequency yet.
    #             # TODO: fix this.
    #             warnings.warn(f'No triggering method for {frequency}kHz yet')
    #             trigger = 0
    #
    #         else:
    #             triggers = DataChopper.find_trigger(data_intp, samples_per_trace=500)
    #             trigger = int(np.median(triggers))
    #
    #     else:
    #         trigger = 0
    #
    #     if interpolated:
    #         intp_traces = DataChopper.chop_traces(data_intp, samples_per_trace=500, trigger=trigger)
    #         return intp_traces
    #
    #     else:
    #         # give back the un-interpolated data.
    #         samples_per_trace = int(5e4 / frequency)
    #         intp_samples = DataChopper.chop_traces(ex_samples, samples_per_trace=500, trigger=trigger)
    #
    #         num_rows = len(data_raw)
    #         num_traces_per_row = len(intp_samples)
    #
    #         traces = np.zeros((num_rows, samples_per_trace * num_traces_per_row))
    #
    #         for i, s in enumerate(intp_samples):
    #             integer_s = np.arange(ceil(s[0]), int(s[-1]) + 1)
    #             integer_s = integer_s[:samples_per_trace]  # trim to the same length
    #
    #             traces[:, i * samples_per_trace : (i+1) * samples_per_trace] = data_raw[:, integer_s]
    #
    #         traces = traces.reshape((num_rows * num_traces_per_row, samples_per_trace))
    #
    #         return traces

