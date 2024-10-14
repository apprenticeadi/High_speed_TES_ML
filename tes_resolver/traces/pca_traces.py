import numpy as np

from .traces import Traces

class PCATraces(Traces):

    def __init__(self, rep_rate, data, labels=None):
        self.rep_rate = rep_rate
        self.freq_str = f'{rep_rate}kHz'

        data = np.atleast_2d(data)  # factor scores
        self._data = data
        if labels is None:
            self._labels = np.full((len(self.data),), -1)
        else:
            self._labels = labels

