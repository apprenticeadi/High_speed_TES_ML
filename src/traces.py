import numpy as np
import matplotlib.pyplot as plt
import copy

from src.fitting_hist import fitting_histogram

class Traces:

    def __init__(self, frequency, data, multiplier=0.6, num_bins=1000):

        self.multiplier = multiplier
        self.num_bins = num_bins

        self.data = data

        self.frequency = frequency

        if frequency == 1000:
            self.freq_str = '1MHz'
        else:
            self.freq_str = f'{frequency}kHz'

        self.idealSamples = 5e4 / frequency  # number of sampling data points per trace
        self.period = int(self.idealSamples)  # integer number of sampling data points per trace (i.e. period of trace)

        self.min_voltage = np.amin(self.data)
        self.max_voltage = np.amax(self.data)

        self.ymin = 5000 * (self.min_voltage // 5000)
        self.ymax = 5000 * (self.max_voltage // 5000 + 1)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    def set_num_bins(self, num_bins):
        self.num_bins = num_bins

    def plot_traces(self, num_traces, x_max=None, fig_name = '', plt_title=''):

        if fig_name == '':
            fig_name = f'{self.freq_str} first {num_traces} traces'

        if x_max == None:
            x_max = self.period

        plt.figure(fig_name)
        for i in range(num_traces):
            plt.plot(self.data[i])
        plt.ylabel('voltage')
        plt.xlabel('time (in sample)')
        plt.xlim(0, x_max)
        plt.ylim(self.ymin, self.ymax)

        if plt_title == '':
            plt_title = f'First {num_traces} traces of {self.freq_str}'
        plt.title(plt_title)

    def average_trace(self, plot=False, fig_name='', plt_title=''):
        """
        :return: Overall average trace of the data
        """

        ave_trace = np.mean(self.data, axis=0)
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

        return ave_trace

    def overlaps(self):

        ave_trace = self.average_trace()
        overlaps = ave_trace @ self.data.T

        return overlaps

    def raw_histogram(self, plot=True, fig_name='', plt_title=''):
        overlaps = self.overlaps()

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
        binning_index, binning_traces = hist_fit.trace_bin(self.data)

        return binning_index, binning_traces

    def characteristic_traces_pn(self, plot=False, fig_name='', plt_title=''):
        """
        :return: Average trace for each photon number
        """

        _, binning_traces = self.bin_traces(plot=False)

        characteristic_traces = np.zeros((len(binning_traces.keys()), self.data.shape[1]), dtype=np.float64)

        for pn in binning_traces.keys():
            characteristic_traces[pn] = np.mean(binning_traces[pn], axis=0)

        if plot:
            if fig_name == '':
                fig_name = f'{self.freq_str} characteristic traces per photon number'
            plt.figure(fig_name)
            for pn in range(characteristic_traces.shape[0]):
                plt.plot(characteristic_traces[pn], label=f'{pn} photons')
                plt.ylabel('voltage')
                plt.xlabel('time (in sample)')
                plt.legend()

                if plt_title == '':
                    plt_title = f'{self.freq_str} characteristic traces'

                plt.title(plt_title)

        return characteristic_traces


    def subtract_offset(self):
        '''
        Ruidi has three methods for subtracting offsets:
        1. In his method outline, he said he subtracts the mean 0-photon trace from every trace, so that the mean
        0-photon trace is all zero. But I don't think this will work for higher frequency traces that have overlapping
        tails.
        2. In his code for 100kHz, he subtracts the minimum value of the average trace (of all photon traces) from every
        trace.
        3. In his code for 600kHz, he subtracts the minimum value of the average 0-photon trace from every trace.

        Here I'm going to try a slightly different 4-th method: subtract the average value of the average 0-photon trace
        from every trace. The reasoning is, the whole point of offset subtraction is such that 0 photon should on
        average incur 0 voltage.
        '''

        binning_index_600, binning_traces_600 = self.bin_traces()

        offset = np.mean(binning_traces_600[0])  # voltage mean value of zero photon traces

        data_shifted = self.data - offset
        self.data = data_shifted
        self.ymax = self.ymax - offset
        self.ymin = self.ymin - offset

        return offset, data_shifted







