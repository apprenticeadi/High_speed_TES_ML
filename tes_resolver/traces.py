import warnings
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import copy
from math import ceil, floor

from tes_resolver.data_chopper import DataChopper


# No classifying algorithms here.

class Traces(object):

    def __init__(self, rep_rate, data, labels=None, sampling_rate=5e4, parse_data=True,
                 trigger_delay: Union[str, int] = 'automatic'):
        """
        Object to handle tes voltage traces. No plotting functionality.
        :param rep_rate: repetition rate, unit: kHz.
        :param data: np.array, each row is assumed to be a trace
        :param sampling_rate: sampling rate, i.e. how many sampling datapoints per second. Default is 50MHz, which
        corresponds to 500 datapoints per period for 100kHz data.
        :param parse_data: If True, parse the data such that each row is a trace and with same length. If False, then
        leave data as is.
        :param trigger_delay: Relevant when parse_data is True. Find the number of samples to delay the trigger,
         such that each trace starts with its rising edge. If 'automatic', then trigger delay is found automatically by
         finding the troughs of the data, which only works for data above 300kHz with significant overlap.
         Otherwise, trigger_delay is an integer value.
        """
        self.rep_rate = rep_rate
        self.freq_str = f'{rep_rate}kHz'

        self.sampling_rate = sampling_rate
        self.ideal_samples = sampling_rate / rep_rate  # the ideal number of sampling data points per trace
        self.period = int(self.ideal_samples)  # integer number of sampling data points per trace (i.e. period of trace)

        # parse data if necessary
        data = np.atleast_2d(data)

        # if automatic trigger delay
        if trigger_delay == 'automatic':
            if rep_rate <= 300:
                trigger_delay = 0
            else:
                trigger_delay = DataChopper.find_trigger(data, samples_per_trace=int(sampling_rate / rep_rate))

        if parse_data:
            if data.shape[1] <= self.period:
                warnings.warn(f'Input data array length <= period={self.period}, no parsing performed. ')
            elif labels is not None:
                # Input labels is not None, labels and data will only be chopped to dimension. There will be no parsing
                data, labels = DataChopper.chop_labelled_traces(data, labels, samples_per_trace=self.period,
                                                                trigger_delay=trigger_delay)
            else:
                # When ideal_samples != period, e.g. 600kHz, ideally some traces should have 84 samples, while others 83
                # samples. What parsing does is remove the extra sample from every 84-long traces, while making sure
                # that every trace still starts roughly at the same relative position.
                data = TraceUtils.parse_data(self.rep_rate, data_raw=data, sampling_rate=self.sampling_rate,
                                             trigger_delay=trigger_delay)

        self._data = data
        if labels is None:
            self._labels = np.full((len(self.data),), -1)
        else:
            self._labels = labels

    @property
    def data(self):
        return copy.deepcopy(self._data)

    @data.setter
    def data(self, new_data):
        self._data = copy.deepcopy(new_data)

    @property
    def labels(self):
        return copy.deepcopy(self._labels)

    @labels.setter
    def labels(self, new_labels):
        self._labels = new_labels

    @property
    def num_traces(self):
        return len(self.data)

    def average_trace(self):
        return np.mean(self.data, axis=0)

    def std_trace(self):
        return np.std(self.data, axis=0)

    def bin_traces(self):
        max_pn = np.max(self.labels)
        pns = np.arange(max_pn + 1)

        # Initialise a dictionary to store indices for each photon number
        indices_dict = {}
        for pn in pns:
            indices_dict[pn] = np.where(self.labels == pn)[0]

        # traces_dict = {}
        # for pn in pns:
        #     traces_dict[pn] = self.data[indices_dict[pn]]

        return indices_dict

    def pn_distribution(self, normalised=False):
        indices_dict = self.bin_traces()

        labels = np.asarray(list(indices_dict.keys()))
        counts = np.zeros_like(labels)
        for i_pn, pn in enumerate(labels):
            counts[i_pn] = len(indices_dict[pn])

        if normalised:
            counts = counts / np.sum(counts)

        return labels, counts

    def pn_traces(self, pn):
        """Return all traces with the specified photon number labels"""
        indices_dict = self.bin_traces()

        if pn not in indices_dict.keys():
            return None
        else:
            return self.data[indices_dict[pn]]

    def characteristic_traces(self):
        indices_dict = self.bin_traces()

        char_traces_dict = {}
        for pn in indices_dict.keys():
            char_traces_dict[pn] = np.mean(self.pn_traces(pn), axis=0)

        return char_traces_dict

    def pca_cleanup(self, pca_components=1):
        data = self.data
        # To perform PCA, first zero the mean along each column
        col_means = np.mean(data, axis=0)
        data_zeroed = data - col_means

        # Singular value decomposition to find factor scores and loading matrix
        P, Delta, QT = np.linalg.svd(data_zeroed, full_matrices=False)
        F = P * Delta  # Factor scores

        '''
        Truncate at first few principal components
        '''
        F_truncated = F[:, :pca_components]
        data_cleaned = F_truncated @ QT[:pca_components, :] + col_means

        self.data = data_cleaned

    def find_offset(self):
        '''Find the median voltage value of the zero-photon traces'''
        zero_traces = self.pn_traces(0)

        if zero_traces is None:
            raise Exception('No 0-photon traces in this dataset. ')
        else:
            return np.median(zero_traces)


class TraceUtils:

    @staticmethod
    def parse_data(rep_rate, data_raw, sampling_rate=5e4, interpolated=False, trigger_delay=0):
        """
        Return numpy array, where each row is a trace. Parsing uses interpolation as an intermediate step, to remove
        the shift in traces due to the cutoff from ideal_samples to period.

        :param rep_rate: Repetition rate (kHz)
        :param data_raw: raw data array
        :param sampling_rate: Sampling rate (kHz).
        :param interpolated: If interpolated is True, then result will contain 500 samples per trace, which contains
        interpolated data points that supplements the original data_raw.
        If False, then int(sampling_rate/frequency) samples per trace.
        :param trigger_delay: Integer value for trigger delay
        :return: Numpy array, where each row is a trace and every trace has the same length (trimmed if necessary)
        """

        ideal_samples = sampling_rate / rep_rate  # not always an integer
        period = int(ideal_samples)  # always an integer with some cutoff.

        # interpolate the data, such that each trace has the same samples as 100kHz data
        f = rep_rate // 100

        # extended samples gives the index of a row. integer indices are the original data from data_raw,
        # decimal indices are the interpolated datapoints.
        # data_interpolated is the data after interpolation, which retains the row number of data_raw
        extended_samples, data_interpolated = DataChopper.interpolate_data(data_raw, f)

        if interpolated:
            # give back the interpolated data
            data_to_chop = data_interpolated
            samples_per_trace = 500

        else:
            # give back the un-interpolated data
            num_rows = len(data_raw)

            # intp_samples is a numpy array of indices, where each row corresponds to a trace, and in each row,
            # integer indices mark the original data from data_raw, whereas decimal ones are the interpolated datapoints.
            # Number of rows in intp_samples is the number of traces per row in data_raw
            intp_samples = DataChopper.chop_traces(extended_samples, samples_per_trace=500,
                                                   trigger_delay=0)  # no triggering, because these are just indices.
            num_traces_per_row = len(intp_samples)

            data_to_chop = np.zeros((num_rows, period * num_traces_per_row))
            for i, s in enumerate(intp_samples):
                integer_s = np.arange(ceil(s[0]), floor(
                    s[-1] + 1))  # the integer indices, which correspond to original data from data_raw
                integer_s = integer_s[:period]  # trim to the same length

                data_to_chop[:, i * period: (i + 1) * period] = data_raw[:, integer_s]

            samples_per_trace = period

        # '''Find trigger'''
        # trigger = str(trigger)
        # if trigger.isdecimal():
        #     trigger = int(trigger)
        #     if trigger < 0:
        #         raise ValueError(f'Negative trigger {trigger} not accepted. ')
        # else:
        #     if trigger == 'automatic':
        #         trigger = 'troughs'
        #     trigger = DataChopper.find_trigger(data_to_chop, samples_per_trace=samples_per_trace, method=trigger)

        data_traces = DataChopper.chop_traces(data_to_chop, samples_per_trace=samples_per_trace,
                                              trigger_delay=trigger_delay)

        return data_traces
