import numpy as np
import os
import pickle
import copy
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from tes_resolver.classifier.classifier import Classifier
from tes_resolver.traces import Traces
import tes_resolver.config as config

class InnerProductClassifier(Classifier):

    def __init__(self, modeltype='IP', multiplier=1., num_bins=1000, target_trace=None, inner_prod_bins=None):
        """Classifier that classifies traces according to their inner product with the average trace. """

        if multiplier > 1 :
            raise ValueError(f'Multiplier {multiplier} is larger than 1')

        super().__init__(modeltype)

        self._params.update({
            'multiplier': multiplier,
            'num_bins': num_bins,
            'target_trace': target_trace,
            'inner_prod_bins': inner_prod_bins,
            # 'time_stamp': config.time_stamp,
        })

        self.default_dir = os.path.join(config.home_dir, 'saved_classifiers', 'InnerProductClassifier')

    # @property
    # def time_stamp(self):
    #     return self._params['time_stamp']

    @property
    def target_trace(self):
        target_trace = self._params['target_trace']
        return copy.deepcopy(target_trace)

    @target_trace.setter
    def target_trace(self, new_target_trace):
        self._params['target_trace'] = copy.deepcopy(new_target_trace)

    @property
    def inner_prod_bins(self):
        """A dictionary. The keys are PN labels, values are the upper limit (inclusive) of inner product for the trace to be classified as this label."""
        inner_prod_bins = self._params['inner_prod_bins']
        if inner_prod_bins is None:
            return {}
        else:
            return copy.copy(inner_prod_bins)

    @inner_prod_bins.setter
    def inner_prod_bins(self, new_inner_prod_bins):
        self._params['inner_prod_bins'] = copy.copy(new_inner_prod_bins)

    @property
    def num_bins(self):
        return self._params['num_bins']

    @property
    def multiplier(self):
        return self._params['multiplier']

    def train(self, trainingTraces: Traces):
        """Train the classifier on the training traces. Updates the target trace and inner_prod_bins."""

        '''Calculate the inner products and fit histogram'''
        self.target_trace = trainingTraces.average_trace()
        overlaps = self.calc_inner_prod(trainingTraces)

        mid_bins, heights = InnerProductUtils.raw_histogram(overlaps, num_bins=self.num_bins)

        popt, _ = InnerProductUtils.fit_histogram(mid_bins, heights)

        '''Find peaks and troughs of histogram'''
        x = np.linspace(min(mid_bins), max(mid_bins), 10000)
        fit = InnerProductUtils.fit_func(x, *popt)

        p, _ = find_peaks(fit)  # positions of peaks
        t, _ = find_peaks(-fit)  # positions of troughs

        '''Identify pn peaks'''
        inner_product_bins = {}
        end_of_identifiable = np.min(overlaps)
        while end_of_identifiable < np.max(overlaps):
            pn = len(inner_product_bins.keys())

            if pn == 0:
                # first peak corresponding to pn=0
                assert t[pn] > p[pn]
                upper = x[t[pn]]

            elif pn < len(p) - 1:
                # these peaks have troughs on both sides.
                assert t[pn - 1] < p[pn] < t[pn]
                upper = x[t[pn]]

            elif pn == len(p) - 1:
                # pn = max - 1. final peak that is identified by find_peak. trough only to the lower side.
                assert t[pn - 1] < p[pn]
                mu = x[p[pn]]
                actual_width = mu - x[t[pn - 1]]
                upper = mu + actual_width

            elif pn == len(p):
                # manually identify a peak # TODO: does this work well?
                es_lower = end_of_identifiable

                remaining_overlaps = overlaps[np.where(overlaps >= es_lower)]
                rem_num_bins = len(remaining_overlaps) // 5
                if rem_num_bins < 1 :
                    rem_num_bins = 1
                rem_heights, rem_bins = np.histogram(remaining_overlaps, bins=rem_num_bins)
                rem_troughs, _ = find_peaks(- rem_heights)

                if len(rem_troughs) >= 1:
                    # there is a visible maximum
                    upper = rem_bins[rem_troughs[0] + 1]
                else:
                    # there is no visible maximum
                    upper = np.max(overlaps)

            else:
                upper = np.max(overlaps)

            end_of_identifiable = upper

            inner_product_bins[pn] = upper
            self.inner_prod_bins = inner_product_bins

    def save(self, filename=None, filedir=None):

        if filename is None:
            filename = self.time_stamp + r'.pkl'
        elif filename[-4:] != r'.pkl':
            filename = filename + r'.pkl'

        if filedir is None:
            filedir = self.default_dir
        os.makedirs(filedir, exist_ok=True)

        fullfilename = os.path.join(filedir, filename)

        with open(fullfilename, 'wb') as output_file:
            pickle.dump(self._params, output_file)

    def load(self, filename, filedir=None):

        if filename[-4:] != r'.pkl':
            filename = filename + r'.pkl'

        if filedir is None:
            filedir = self.default_dir

        fullfilename = os.path.join(filedir, filename)

        with open(fullfilename, 'rb') as input_file:
            params = pickle.load(input_file)

        self._params.update(params)

    def predict(self, unknownTraces: Traces, update=False):
        overlaps = self.calc_inner_prod(unknownTraces)  # at least one dimensional
        photon_bins = np.array(list(self.inner_prod_bins.values()))

        if self.multiplier == 1:
            labels = np.digitize(overlaps, photon_bins, right=True)  # overlaps > np.max(photon_bins) are identified as max(pn)+1

        else:
            if len(photon_bins) <= 1:
                raise ValueError('Not enough photon number bins to apply non-unity multiplier.')

            scaled_bins = np.zeros(2*len(photon_bins))
            halfwidths = np.diff(photon_bins) / 2
            midbins = (photon_bins[1:] + photon_bins[:-1]) / 2

            lowers = midbins - self.multiplier * halfwidths
            uppers = midbins + self.multiplier * halfwidths\

            scaled_bins[1:-1] = np.stack((lowers, uppers), axis=1).flatten()

            scaled_bins[0] = 2 * photon_bins[0] - scaled_bins[1]
            scaled_bins[-1] = 2 * photon_bins[-1] - scaled_bins[-2]

            raw_labels = np.digitize(overlaps, scaled_bins)
            raw_labels[raw_labels % 2 != 0] = -1  # sit between bins and is rendered invalid

            labels = raw_labels // 2

        if update:
            unknownTraces.labels = labels

        return labels

    def calc_inner_prod(self, unknownTraces: Traces):
        trace_data = unknownTraces.data
        trace_data = np.atleast_2d(trace_data)

        overlaps = self.target_trace @ trace_data.T

        return overlaps



class InnerProductUtils:

    @staticmethod
    def fit_func(x, *params):
        '''Function used to fit the histogram (sum of multiple Gaussians) '''
        y = np.zeros_like(x)
        for i in range(0, len(params) - 1, 3):
            mu = params[i]
            amp = params[i + 1]
            sig = params[i + 2]
            y = y + amp * np.exp(-((x - mu) / abs(sig)) ** 2)
        return y

    @staticmethod
    def guess_peaks(mid_bins, heights, kernel_size=10):
        """Guess peak positions for the histogram. This is used as guess values for fitting the muli-Gaussian curve to
        this histogram. Histogram defined by heights and positions (mid_bin) of each bin. """

        kernel = np.ones(kernel_size) / kernel_size
        data_convolved_10 = np.convolve(heights, kernel, mode='same')

        peaks, _ = find_peaks(data_convolved_10, height=2, distance=20, prominence=1)
        amplitudes = data_convolved_10[peaks]

        positions = mid_bins[peaks]

        return positions, amplitudes

    @staticmethod
    def raw_histogram(overlaps, num_bins):
        heights, bin_edges = np.histogram(overlaps, bins=num_bins)
        mid_bins = (bin_edges[1:] + bin_edges[:-1]) / 2

        return mid_bins, heights

    @staticmethod
    def fit_histogram(mid_bins, heights):
        """Fit the multi-gaussian function to the histogram of overlaps"""

        '''Initial guess for the fit. '''
        positions, amplitudes = InnerProductUtils.guess_peaks(mid_bins, heights)
        num_peaks = len(positions)
        sigmas = np.diff(positions) / 4
        sigmas = np.append(sigmas, sigmas[-1])
        guess = np.stack((positions, amplitudes, sigmas), axis=1).flatten()

        '''Constrained optimisation for fit'''
        bounds = (
            [min(mid_bins), min(heights), 0.] * num_peaks,
            [max(mid_bins), max(heights), max(sigmas)] * num_peaks
        )
        # popt contains [position, amplitude, sigma] of each Gaussian peak
        popt, pcov = curve_fit(InnerProductUtils.fit_func, mid_bins, heights, p0=guess, bounds=bounds)

        return popt, pcov
