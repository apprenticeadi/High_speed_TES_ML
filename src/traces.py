import warnings

import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm.auto import tqdm
from src.fitting_hist import fitting_histogram

class Traces:

    def __init__(self, frequency, data, multiplier=1., num_bins=1000):

        self._multiplier = multiplier
        self._num_bins = num_bins

        self._data = data

        self._frequency = frequency

        # no longer used in the new conventions
        # if frequency == 1000:
        #     self.freq_str = '1MHz'
        # else:
        #     self.freq_str = f'{frequency}kHz'
        self.freq_str = f'{frequency}kHz'

        self.idealSamples = 5e4 / frequency  # number of sampling data points per trace
        self.period = int(self.idealSamples)  # integer number of sampling data points per trace (i.e. period of trace)

        self.min_voltage = np.amin(data)
        self.max_voltage = np.amax(data)

        # self.min_voltage = min(data.any())
        # self.max_voltage = max(data.any())

        self.ymin = 5000 * (self.min_voltage // 5000)
        self.ymax = 5000 * (self.max_voltage // 5000 + 1)

    @property
    def multiplier(self):
        return self._multiplier

    @multiplier.setter
    def multiplier(self, new_multiplier):
        self._multiplier = new_multiplier

    @property
    def num_bins(self):
        return self._num_bins

    @num_bins.setter
    def num_bins(self, new_num_bins):
        self._num_bins = new_num_bins

    @property
    def frequency(self):
        return self._frequency

    def get_data(self):
        """numpy array is mutable"""
        return copy.deepcopy(self._data)

    def guess_peak(self):
        return self.period // 3

    def plot_traces(self, num_traces, x_max=None, fig_name = '', plt_title=''):
        if fig_name == '':
            fig_name = f'{self.freq_str} first {num_traces} traces'

        if x_max == None:
            x_max = self.period

        plt.figure(fig_name)
        for i in range(num_traces):
            plt.plot(self._data[i])
        plt.ylabel('voltage')
        plt.xlabel('time (in sample)')
        plt.xlim(0, x_max)
        plt.ylim(self.ymin, self.ymax)

        if plt_title == '':
            plt_title = f'First {num_traces} traces of {self.freq_str}'
        plt.title(plt_title)

    def plot_trace_trains(self, num_trains, num_traces, x_max=None, fig_name='', plt_title=''):
        if fig_name == '':
            fig_name = f'{self.freq_str} first {num_trains} trace trains'

        if x_max == None:
            x_max = self.period * num_traces

        plt.figure(fig_name)
        for i in range(num_trains):
            train = self._data[i*num_traces: (i+1)*num_traces, :].flatten()
            plt.plot(train)
        plt.ylabel('voltage')
        plt.xlabel('time (in sample)')
        plt.xlim(0, x_max)
        plt.ylim(self.ymin, self.ymax)

        if plt_title == '':
            plt_title = f'First {num_trains} trace trains of {self.freq_str}'
        plt.title(plt_title)


    def average_trace(self, plot=False, fig_name='', plt_title=''):
        """
        :return: Overall average trace of the data
        """

        ave_trace = np.mean(self._data, axis=0)
        std_trace = np.std(self._data, axis = 0)
        if plot:
            if fig_name == '':
                fig_name = f'{self.freq_str} average trace'

            plt.figure(fig_name)
            plt.plot(ave_trace)
            plt.plot(ave_trace + std_trace, linestyle='dashed')
            plt.plot(ave_trace - std_trace, linestyle='dashed')
            plt.ylabel('voltage')
            plt.xlabel('time (in sample)')
            plt.ylim(self.ymin, self.ymax)

            if plt_title == '':
                plt_title = f'Average trace of {self.freq_str}'
            plt.title(plt_title)

        return ave_trace, ave_trace+std_trace, ave_trace - std_trace

    def inner_products(self):

        ave_trace,stdp,stdm = self.average_trace()
        overlaps = ave_trace @ self._data.T

        return overlaps

    def raw_histogram(self, plot=True, fig_name='', plt_title=''):
        overlaps = self.inner_products()

        if plot:

            if fig_name == '':
                fig_name = f'{self.freq_str} raw stegosaurus'
            plt.figure(fig_name)
            heights, bin_edges, _ = plt.hist(overlaps, bins=self.num_bins, color='aquamarine')
            plt.xlabel('overlaps')

            if plt_title == '':
                plt_title = f'{self.freq_str} raw stegosaurus'
            plt.title(plt_title)
        else:
            heights, bin_edges = np.histogram(overlaps, bins=self.num_bins)

        return overlaps, heights, bin_edges

    def fit_histogram(self, plot=False, fig_name=''):

        overlaps, heights, bin_edges = self.raw_histogram(plot=False)
        mid_bins = (bin_edges[1:] + bin_edges[:-1]) / 2

        hist_fit = fitting_histogram(heights, mid_bins, overlaps, self.multiplier)

        if plot:
            if fig_name == '':
                fig_name = f'{self.freq_str} fit stegosaurus'
            lower_list, upper_list = hist_fit.fitting(plot=plot, fig_name=fig_name)

        return hist_fit

    def bin_traces(self, plot=False, fig_name=''):

        hist_fit = self.fit_histogram(plot=plot, fig_name=fig_name)
        binning_index, binning_traces = hist_fit.trace_bin(self.get_data())

        return binning_index, binning_traces

    def pn_bar_plot(self,plot = True, fig_name='', plt_title='', normalised=False):

        binning_index, _ = self.bin_traces(plot=False)

        pns = binning_index.keys()
        counts = [len(binning_index[pn]) for pn in pns]
        counts = np.asarray(counts)
        if normalised:
            counts = counts / np.sum(counts)

        if plot:
            if fig_name == '':
                if normalised:
                    fig_name = f'Normalised {self.freq_str} photon number bar plot'
                else:
                    fig_name = f'{self.freq_str} photon number bar plot'
            if plt_title == '':
                plt_title = f'{self.freq_str} photon number distribution for multiplier={self.multiplier}'

            plt.figure(fig_name)
            plt.bar(pns, counts)
            plt.xlabel('Photon number')
            if normalised:
                plt.ylabel('Probability')
            else:
                plt.ylabel('Counts')
            plt.title(plt_title)

        return pns, counts

    def characteristic_traces_pn(self, plot=False, fig_name='', plt_title=''):
        """
        :return: Average trace for each photon number
        """

        _, binning_traces = self.bin_traces(plot=False)

        characteristic_traces = np.zeros((len(binning_traces.keys()), self._data.shape[1]), dtype=np.float64)
        char_traces_err = np.zeros((len(binning_traces.keys()), self._data.shape[1]), dtype=np.float64)

        for pn in binning_traces.keys():
            characteristic_traces[pn] = np.mean(binning_traces[pn], axis=0)
            char_traces_err[pn] = np.std(binning_traces[pn], axis = 0)

        #colours to ensure plot is clean
        colours = ['r','b', 'g', 'c', 'm', 'y', 'k', 'r', 'b','g']

        if plot:
            if fig_name == '':
                fig_name = f'{self.freq_str} characteristic traces per photon number'
            plt.figure(fig_name)
            for pn in range(characteristic_traces.shape[0]):
                plt.plot(characteristic_traces[pn], label=f'{pn} photons')
                plt.plot(characteristic_traces[pn]+char_traces_err[pn],linestyle = 'dashed')
                plt.plot(characteristic_traces[pn]-char_traces_err[pn],linestyle = 'dashed')
                plt.ylabel('voltage')
                plt.xlabel('time (in sample)')
                plt.legend()

                if plt_title == '':
                    plt_title = f'{self.freq_str} characteristic traces'

                plt.title(plt_title)

        return characteristic_traces
    def total_traces(self):
        '''
        returns all possible traces, ie average trace with std error trace
        '''
        _, binning_traces = self.bin_traces(plot=False)

        characteristic_traces = np.zeros((len(binning_traces.keys()), self._data.shape[1]), dtype=np.float64)
        char_traces_err = np.zeros((len(binning_traces.keys()), self._data.shape[1]), dtype=np.float64)

        for pn in binning_traces.keys():
            characteristic_traces[pn] = np.mean(binning_traces[pn], axis=0)
            char_traces_err[pn] = np.std(binning_traces[pn], axis = 0)


        char_err_max = characteristic_traces + char_traces_err
        char_err_min = characteristic_traces - char_traces_err
        total_traces = np.concatenate((char_err_min, characteristic_traces, char_err_max))
        return total_traces

    def abs_voltage_diffs(self):
        """
        Bin the traces. And for each trace, calculate its sum(|voltage difference|) from the corresponding photon number
        characteristic trace
        """
        binning_index, binning_traces = self.bin_traces()
        char_traces = self.characteristic_traces_pn()
        diff = {}
        for pn in binning_traces.keys():
            diff[pn] = np.sum(np.abs(binning_traces[pn] - char_traces[pn]), axis=1)

        return diff

    #TODO: what is a good way of subtracting offset?
    def subtract_offset(self):
        '''
        Ruidi has three methods for subtracting offsets:
        1. In his method outline, he said he subtracts the mean 0-photon trace from every trace, so that the mean
        0-photon trace is all zero. But I don't think this will work for higher frequency traces that have overlapping
        tails.
        2. In his code for 100kHz, he subtracts the minimum value of the average trace (of all photon traces) from every
        trace.
        3. In his code for 600kHz, he subtracts the minimum value of the average 0-photon trace from every trace.

        Deciding the offset according to 0-photon trace is probably not a good idea. Because for higher rep rates, there
        is no good way of deciding what are 0-photon traces.
        '''

        # if self.frequency == 100:
        #     offset = np.min(self.average_trace())
        # else:
        #     binning_index, binning_traces = self.bin_traces()
        #     offset = np.min(np.mean(binning_traces[0], axis=0))  # voltage mean value of zero photon traces

        binning_index, binning_traces = self.bin_traces()
        offset = np.mean(np.mean(binning_traces[0], axis=0))  # voltage mean value of zero photon traces

        # offset = np.min(self.average_trace())

        data_shifted = self.get_data() - offset
        self._data = data_shifted
        self.ymax = self.ymax - offset
        self.ymin = self.ymin - offset

        return offset, self.get_data()

    def overlap_to_high_freq(self, high_frequency, num_traces=0):
        if high_frequency <= self.frequency:
            raise ValueError(f'New frequency {high_frequency}kHz is lower than current frequency {self.frequency}')

        current_data = self.get_data()
        if num_traces==0:
            num_traces = current_data.shape[0]
            #num_traces = len(current_data)

        new_period = int(5e4 / high_frequency)  # truncated.
        # old_period = 500
        data_overlapped = np.zeros(new_period * (num_traces - 1) + self.period)

        for i in range(num_traces):
            data_overlapped[i*new_period: i*new_period + self.period] += current_data[i, :]

        return data_overlapped[: new_period * num_traces].reshape((num_traces, new_period))

    def generate_high_freq_data(self,frequency):
        data_100 = self.get_data()
        num_traces = data_100.shape[0]
        new_period = int(5e4/frequency)  # truncated, note the truncation error.
        overlapped = np.zeros(num_traces-1)
        overlapped = []
        for i in range(2,num_traces):
            new_peak = data_100[i][:new_period]
            old_tail = data_100[i-1][new_period:2*new_period]
            if frequency>300:
                prev_two = data_100[i-2][2*new_period:3*new_period]
                overlapped.append(new_peak+old_tail+prev_two)
            else:
                overlapped.append(new_peak + old_tail)

        return np.array(overlapped)


    def pca_cleanup(self, num_components=1):
        data = self.get_data()
        # To perform PCA, first zero the mean along each column
        col_means = np.mean(data, axis=0)
        data_zeroed = data - col_means

        # Singular value decomposition to find factor scores and loading matrix
        P, Delta, QT = np.linalg.svd(data_zeroed, full_matrices=False)
        F = P * Delta  # Factor scores

        '''
        Truncate at first few principal components
        '''
        F_truncated = F[:, :num_components]
        data_cleaned = F_truncated @ QT[:num_components, :] + col_means

        return data_cleaned

    def return_labelled_traces(self):
        '''
        returns the labels at the correct indices
        '''
        binning_index, _ = self.bin_traces(plot=False)
        keys = [key for key in binning_index]
        values = list(binning_index.values()) # [binning_index[key] for key in binning_index]
        # indices = np.zeros(len(self._data))
        indices = np.full((len(self._data)), -1)

        for i in range(len(keys)):
            for j in values[i]:
                indices[j] = keys[i]

        return indices

    def return_av_diff(self):
        # TODO: suspicious, why average over axis=1?
        warnings.warn('Function suspicious, double check. ')
        ave_trace = np.mean(self._data, axis=1)
        return np.min(ave_trace)


class IntpTraces(Traces):
    # TODO: a class that treats interpolated traces. The period and ideal number of samples need to change.


    pass





