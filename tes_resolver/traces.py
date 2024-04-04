import warnings

import numpy as np
import matplotlib.pyplot as plt
import copy

from tes_resolver.data_chopper import DataChopper

# No classifying algorithms here.

class Traces(object):

    def __init__(self, rep_rate, data, labels=None, sampling_rate=5e4):
        """
        Object to handle tes voltage traces. No plotting functionality.
        :param rep_rate: repetition rate, unit: kHz.
        :param data: array_like, each element is assumed to be a trace
        :param sampling_rate: sampling rate, i.e. how many sampling datapoints per second. Default is 50MHz, which
        corresponds to 500 datapoints per period for 100kHz data.
        """

        #TODO: maybe make it possible to change the unit.
        self.rep_rate = rep_rate
        self.freq_str = f'{rep_rate}kHz'

        self._data = data
        if labels is None:
            self._labels = np.full((len(self.data), ), np.nan)
        else:
            if len(labels) != len(self.data):
                raise ValueError('Input labels and data dimensions do not match')
            else:
                self._labels = labels

        self.sampling_rate = sampling_rate
        self.ideal_samples = sampling_rate / rep_rate  # the ideal number of sampling data points per trace
        self.period = int(self.ideal_samples)  # integer number of sampling data points per trace (i.e. period of trace)


    @property
    def data(self):
        return copy.deepcopy(self._data)

    @property
    def labels(self):
        return copy.deepcopy(self._labels)

    @property
    def num_traces(self):
        return self.data.shape[0]

    def average_trace(self):
        return np.mean(self.data, axis=0)

    def std_trace(self):
        return np.std(self.data, axis=0)

    def inner_products(self, target_trace=None):
        if target_trace is None:
            target_trace = self.average_trace()

        overlaps = target_trace @ self.data.T

        return overlaps

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

        return data_cleaned


