# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:04:55 2022

@author: ruidi zhu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
# %%
class fitting_histogram:
    def __init__(self, heights, midBins, overlap, multiplier):
        # heights and positions of each bins in histogram
        self.heights = heights
        self.midBins = midBins
        self.overlap = overlap
        # define how many sigma of data away from the mean we want to process
        self.multiplier = multiplier

    def finding_peaks(self, plot=True):
        '''
        use data smoothing method to find the peak positions of each bins of the histogram, 
        will be useful for fitting the historgam
        '''
        kernel_size = 10
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved_10 = np.convolve(self.heights, kernel, mode='same')

        plt.plot(self.midBins, data_convolved_10)
        peaks, _ = find_peaks(data_convolved_10, height=2,
                              distance=30, prominence=1, width=10)
        amplitudes = data_convolved_10[peaks]

        #heights,bins,_ = plt.hist(overlap_list,bins=1000,range=(-0.1e10,1.5e10))
        if plot == True:
            plt.plot(self.midBins[peaks], amplitudes, "x")
            plt.plot(np.zeros_like(data_convolved_10), "--", color="gray")
        plt.show()
        return peaks, amplitudes

    def func(self, x, *params):
        '''
        functions used to fit the histogram (sum of multiple Gaussians)
        '''
        y = np.zeros_like(x)
        for i in range(0, len(params)-1, 3):
            mu = params[i]
            amp = params[i+1]
            sig = params[i+2]
            y = y + amp * np.exp(-((x - mu)/abs(sig))**2)
        return y

    def fitting(self):
        '''
        fitting the histogram
        '''
        peaks, amplitudes = self.finding_peaks(plot=True)

        # initial guess for fitting the historgams (calculated from smoothing method defined in finding_peaks)
        positions = self.midBins[peaks]  # peak positions
        sigma = (positions[1] - positions[0]) / 4  # std of each peak
        self.numPeaks = len(peaks)  # number of peaks

        guess = []
        for i in range(self.numPeaks):
            guess += [positions[i], amplitudes[i], sigma]

        popt, pcov = curve_fit(self.func, self.midBins, self.heights, p0=guess)

        sort = np.argsort(popt[::3])

        sorted_popt = []
        for i in sort:
            count = 3*i
            popt[count+2] = abs(popt[count+2])
            print(f'{i}th peak:', popt[count:count+3])
            sorted_popt.append(list(popt[count:count+3]))
        sorted_popt = list(np.array(sorted_popt).flat)

        x = np.linspace(min(self.midBins), max(self.midBins), 10000)
        fit = self.func(x, *sorted_popt)
        plt.figure(figsize=(8, 5))
        plt.hist(self.overlap, bins=1000)
        plt.plot(x, fit)
        plt.xlabel('overlap', size=14)
        plt.ylabel('entries', size=14)

        upper_list = []
        lower_list = []
        for i in range(0, self.numPeaks):
            mu = sorted_popt[i*3]
            width = sorted_popt[i*3+2] * self.multiplier
            upper = mu + width
            lower = mu - width
            plt.vlines(upper, 0, self.func(upper, *sorted_popt), color='red')
            plt.vlines(lower, 0, self.func(lower, *sorted_popt), color='red')
            upper_list.append(upper)
            lower_list.append(lower)
        plt.show()
        self.lower_list = np.sort(lower_list)
        self.upper_list = np.sort(upper_list)
        return self.lower_list, self.upper_list

    def trace_bin(self,data):
        '''
        grouping the traces corresponding to each photon numbers 
        '''
        binning_index = [[i for i in range(len(self.overlap)) if self.lower_list[photon_number] < self.overlap[i] < self.upper_list[photon_number]] for photon_number in range(0, self.numPeaks)]
        binning_traces = [[data[index] for index in index_list]  for index_list in binning_index]
        return binning_index, binning_traces
