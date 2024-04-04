"""
Created on Thu Aug 25 19:04:55 2022

@author: ruidi zhu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import warnings

class HistFit:
    def __init__(self, heights, mid_bins, overlaps):
        # heights and positions of each bins in histogram
        self.heights = heights
        self.mid_bins = mid_bins
        self.overlaps = overlaps
        # define how many sigma of data away from the mean we want to process

        self.num_peaks = np.nan

    def finding_peaks(self, plot=False, ax=None):
        '''
        use data smoothing method to find the peak positions of each bins of the histogram,
        will be useful for fitting the histogram
        '''
        kernel_size = 10
        kernel = np.ones(kernel_size) / kernel_size
        data_convolved_10 = np.convolve(self.heights, kernel, mode='same')

        if plot:
            if ax is None:
                fig, ax = plt.subplots()
            ax.plot(self.mid_bins, data_convolved_10, label='data smoothing')
            ax.plot(self.mid_bins, np.zeros_like(data_convolved_10), "--", color="gray")

        # peaks, _ = find_peaks(data_convolved_10, height=2, distance=30, prominence=1, width=10)
        peaks, _ = find_peaks(data_convolved_10, height=2, distance=20, prominence=1)
        amplitudes = data_convolved_10[peaks]

        if plot:
            ax.plot(self.mid_bins[peaks], amplitudes, "x")
            ax.legend()
        return peaks, amplitudes

    def func(self, x, *params):
        '''
        functions used to fit the histogram (sum of multiple Gaussians)
        '''
        y = np.zeros_like(x)
        for i in range(0, len(params) - 1, 3):
            mu = params[i]
            amp = params[i + 1]
            sig = params[i + 2]
            y = y + amp * np.exp(-((x - mu) / abs(sig)) ** 2)
        return y

    def fitting(self, plot=False, ax=None, coloring=True, indexing=True,
                hist_color='aquamarine', plot_fit=True):
        '''
        fitting the histogram
        '''

        peaks, amplitudes = self.finding_peaks(plot=False)
        self.num_peaks = len(peaks)  # number of peaks

        # initial guess for fitting the histograms (calculated from smoothing method defined in finding_peaks)
        guess = []
        positions = self.mid_bins[peaks]  # peak positions
        sigmas = np.diff(positions) / 4
        sigmas = np.append(sigmas, sigmas[-1])
        for i in range(self.num_peaks):
            guess += [positions[i], amplitudes[i], sigmas[i]]

        # constrained optimisation
        bounds = (
            [min(self.mid_bins), min(self.heights), 0.] * self.num_peaks,
            [max(self.mid_bins), max(self.heights), max(sigmas)] * self.num_peaks
        )
        popt, pcov = curve_fit(self.func, self.mid_bins, self.heights, p0=guess, bounds=bounds)  # popt contains [position, amplitude, sigma] of each Gaussian peak

        # find peaks and troughs of histogram.
        x = np.linspace(min(self.mid_bins), max(self.mid_bins), 10000)
        fit = self.func(x, *popt)

        p, _ = find_peaks(fit)  # positions of peaks
        t, _ = find_peaks(-fit)  # positions of troughs

        if plot:
            if ax is None:
                fig, ax = plt.subplots()

            half_bin_widths = 0.5 * np.diff(self.mid_bins)
            half_bin_widths = np.append(half_bin_widths, half_bin_widths[-1])
            bins = self.mid_bins - half_bin_widths
            bins = np.append(bins, bins[-1] + half_bin_widths[-1])
            ax.hist(self.overlaps, bins=bins, color=hist_color)

            if plot_fit:
                ax.plot(x, fit, 'r-')
                ax.plot(x[p], fit[p], 'x', label='peaks')
                ax.plot(x[t], fit[t], 'x', label='troughs')

            ax.set_xlabel('Inner product')
            ax.set_ylabel('Counts')

        # identify pn peaks
        inner_product_bins = {} # keys are PN labels, values are the upper limit of inner product for the trace to be classified as this label.

        end_of_identifiable = np.min(self.overlaps)
        while end_of_identifiable < np.max(self.overlaps):
            pn = len(inner_product_bins.keys())

            if pn == 0:
                # first peak corresponding to pn=0
                assert t[pn] > p[pn]
                upper = x[t[pn]]

            elif pn < len(p) - 1:
                # these peaks have troughs on both sides.
                assert t[pn-1] < p[pn] < t[pn]
                upper = x[t[pn]]

            elif pn == len(p) - 1:
                # pn = max - 1. final peak that is identified by find_peak. trough only to the lower side.
                assert t[pn-1] < p[pn]
                mu = x[p[pn]]
                actual_width = mu - x[t[pn-1]]
                upper = mu + actual_width

            elif pn == len(p):
                # manually identify a peak # TODO: does this work well?
                es_lower = end_of_identifiable

                remaining_overlaps = self.overlaps[np.where(self.overlaps >= es_lower)]
                rem_heights, rem_bins = np.histogram(remaining_overlaps, bins= len(remaining_overlaps) // 5)
                rem_troughs, _ = find_peaks(- rem_heights)

                if len(rem_troughs) >=1:
                    # there is a visible maximum
                    upper = rem_bins[rem_troughs[0] + 1]
                else:
                    # there is no visible maximum
                    upper = np.max(self.overlaps)

            else:
                upper = np.max(self.overlaps)

            end_of_identifiable = upper

            inner_product_bins[pn] = upper

        self.num_peaks = len(inner_product_bins.keys())
        return inner_product_bins

    def binning(self, multiplier=1.):
        '''
        grouping the traces corresponding to each photon numbers
        '''

        # np digitize is a bad idea when the fit is bad and two photon number bins overlap.
        # photon_bins = np.array([i for j in zip(self.lower_list, self.upper_list) for i in j])
        # bin_indices = np.digitize(self.overlap, photon_bins)  # bin 0 for overlap left of lower[0], bin 1 for photon_num=0, bin 3 for photon_num=1, bin 5 for photon_num=2 etc.
        inner_product_bins = self.fitting(plot=False)

        if multiplier > 1.:
            raise ValueError(f'Multiplier {multiplier} is larger than 1')
        elif multiplier == 1.:
            photon_bins = list(inner_product_bins.values())
            bin_indices = np.digitize(self.overlaps, photon_bins, right=True)  # overlaps > np.max(photon_bins) are identified as max(pn)+1

        else:



        #TODO: just bin the overlaps. don't worry about others.