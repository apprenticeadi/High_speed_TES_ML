import numpy as np
import pickle
import copy

from tes_resolver.classifier.classifier import Classifier
from tes_resolver.traces import Traces


class InnerProductClassifier(Classifier):

    def __init__(self, multiplier=1., num_bins=1000):
        """Classifier that classifies traces according to their inner product with the average trace. """

        self.multiplier = multiplier
        self.num_bins = num_bins

        self._target_trace = None
        self._inner_prod_bins = {}  # keys are PN labels, values are the upper limit of inner product for the trace to be classified as this label.

    @property
    def target_trace(self):
        return copy.deepcopy(self._target_trace)

    @target_trace.setter
    def target_trace(self, new_target_trace):
        self._target_trace = new_target_trace

    def train(self, calTraces: Traces):
        self.target_trace = calTraces.average_trace()
        overlaps = calTraces.inner_products(target_trace=self.target_trace)
        heights, bin_edges = np.histogram(overlaps, bins=self.num_bins)

        mid_bins = (bin_edges[1:] + bin_edges[:-1]) / 2

        #TODO: result should be a set of inner product bins.
        # Keep existing fitting_histogram functions. Don't need to reinvent the wheel.


