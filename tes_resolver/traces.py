import warnings

import numpy as np
import matplotlib.pyplot as plt
import copy
from math import ceil, floor

from tes_resolver.data_chopper import DataChopper

# No classifying algorithms here.


#TODO: 1. Logically, sampling_rate should not be here. 2. Data should always be homogoenous 2d array.
class Traces(object):

    def __init__(self, rep_rate, data, labels=None, sampling_rate=5e4, parse_data=True, **data_parsing_kwargs):
        """
        Object to handle tes voltage traces. No plotting functionality.
        :param rep_rate: repetition rate, unit: kHz.
        :param data: np.array, each row is assumed to be a trace
        :param sampling_rate: sampling rate, i.e. how many sampling datapoints per second. Default is 50MHz, which
        corresponds to 500 datapoints per period for 100kHz data.
        :param parse_data: If True, parse the data such that each row is a trace and with same length. If False, then
        leave data as is.
        """

        #TODO: what if only collect the trace partially?
        self.rep_rate = rep_rate
        self.freq_str = f'{rep_rate}kHz'

        self.sampling_rate = sampling_rate
        self.ideal_samples = sampling_rate / rep_rate  # the ideal number of sampling data points per trace
        self.period = int(self.ideal_samples)  # integer number of sampling data points per trace (i.e. period of trace)

        # TODO: doesn't work on 150kHz data.
        # parse data if necessary
        data = np.atleast_2d(data)
        parse_args = {'interpolated': False, 'trigger': 'automatic'}
        parse_args.update(data_parsing_kwargs)
        if parse_data and data.shape[1] != self.period:
            data, trigger = TraceUtils.parse_data(self.rep_rate, data_raw=data, sampling_rate=self.sampling_rate, **parse_args)
        else:
            trigger = 0

        self._data = data
        self.trigger = trigger
        if labels is None:
            self._labels = np.full((len(self.data), ), -1)
        else:
            if len(labels) != len(self.data):
                raise ValueError('Input labels and data dimensions do not match')
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
        pns = set(self.labels)

        # Initialise a dictionary to store indices for each photon number
        indices_dict = {}
        for pn in pns:
            indices_dict[pn] = np.where(self.labels == pn)[0]

        traces_dict = {}
        for pn in pns:
            traces_dict[pn] = self.data[indices_dict[pn]]

        return indices_dict, traces_dict

    def pn_distribution(self, normalised=False):
        indices_dict, _ = self.bin_traces()

        labels = np.asarray(list(indices_dict.keys()))
        counts = np.zeros_like(labels)
        for i_pn, pn in enumerate(labels):
            counts[i_pn] = len(indices_dict[pn])

        if normalised:
            counts = counts / np.sum(counts)

        return labels, counts

    def characteristic_traces(self):
        _, traces_dict = self.bin_traces()

        char_traces_dict = {}
        for pn in traces_dict.keys():
            char_traces_dict[pn] = np.mean(traces_dict[pn], axis=0)

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
        '''Find the average voltage value of zero-photon traces'''
        _, traces_dict = self.bin_traces()

        if 0 in traces_dict.keys():
            return np.mean(traces_dict[0])
        else:
            raise Exception('No 0-photon traces in this dataset. ')



class TraceUtils:

    @staticmethod
    def parse_data(rep_rate, data_raw, sampling_rate=5e4, interpolated=False, trigger='automatic'):
        """
        Return numpy array, where each row is a trace

        :param rep_rate: Repetition rate (kHz)
        :param data_raw: raw data array
        :param sampling_rate: Sampling rate (kHz).
        :param interpolated: If interpolated is True, then result will contain 500 samples per trace, which contains
        interpolated data points that supplements the original data_raw.
        If False, then int(sampling_rate/frequency) samples per trace.
        :param trigger: If integer value, then specifies the trigger value; if string, then specifies the method to find
        trigger.

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
            intp_samples = DataChopper.chop_traces(extended_samples, samples_per_trace=500, trigger=0)  # no triggering, because these are just indices.
            num_traces_per_row = len(intp_samples)

            data_to_chop = np.zeros((num_rows, period * num_traces_per_row))
            for i, s in enumerate(intp_samples):
                integer_s = np.arange(ceil(s[0]), floor(s[-1] + 1))  # the integer indices, which correspond to original data from data_raw
                integer_s = integer_s[:period]  # trim to the same length

                data_to_chop[:, i* period: (i+1) * period] = data_raw[:, integer_s]

            samples_per_trace = period

        '''Find trigger'''
        trigger = str(trigger)
        if trigger.isdecimal():
            trigger = int(trigger)
            if trigger < 0:
                raise ValueError(f'Negative trigger {trigger} not accepted. ')
        else:
            if trigger == 'automatic':
                trigger = 'troughs'
            trigger = DataChopper.find_trigger(data_to_chop, samples_per_trace=samples_per_trace, method=trigger)

        data_traces = DataChopper.chop_traces(data_to_chop, samples_per_trace=samples_per_trace, trigger=trigger)

        return data_traces, trigger

