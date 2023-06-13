import numpy as np
import matplotlib.pyplot as plt
import copy

from src.fitting_hist import fitting_histogram

class Traces:

    def __init__(self, frequency, data, multiplier=0.6, num_bins=1000):

        self.multiplier = multiplier
        self.num_bins = num_bins

        self._data = data

        self.frequency = frequency

        if frequency == 1000:
            self.freq_str = '1MHz'
        else:
            self.freq_str = f'{frequency}kHz'

        self.idealSamples = 5e4 / frequency  # number of sampling data points per trace
        self.period = int(self.idealSamples)  # integer number of sampling data points per trace (i.e. period of trace)

        self.min_voltage = np.amin(data)
        self.max_voltage = np.amax(data)

        self.ymin = 5000 * (self.min_voltage // 5000)
        self.ymax = 5000 * (self.max_voltage // 5000 + 1)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    def set_num_bins(self, num_bins):
        self.num_bins = num_bins

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

    def pn_bar_plot(self, fig_name='', plt_title=''):

        binning_index, _ = self.bin_traces(plot=False)
        max_pn_steg = max(binning_index.keys())
        num_trace_per_pn = [len(binning_index[pn]) for pn in range(max_pn_steg + 1)]

        if fig_name == '':
            fig_name = f'{self.freq_str} photon number bar plot'
        if plt_title == '':
            plt_title = f'{self.freq_str} photon number bar plot for multiplier={self.multiplier}'

        plt.figure(fig_name)
        plt.bar(list(range(max_pn_steg + 1)), num_trace_per_pn)
        plt.xlabel('Photon number')
        plt.ylabel('Counts')
        plt.title(plt_title)

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
                plt.plot(characteristic_traces[pn], label=f'{pn} photons', color = colours[pn])
                plt.plot(characteristic_traces[pn]+char_traces_err[pn],linestyle = 'dashed', label=f'{pn} photons +1std', color = colours[pn])
                plt.plot(characteristic_traces[pn]-char_traces_err[pn],linestyle = 'dashed', label=f'{pn} photons -1 std', color = colours[pn])
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

        new_period = int(5e4 / high_frequency)
        data_overlapped = np.zeros(new_period * (num_traces - 1) + self.period)
        for i in range(num_traces):
            data_overlapped[i*new_period: i*new_period + self.period] = current_data[i, :]

        return data_overlapped[: new_period * num_traces].reshape((num_traces, new_period))

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


