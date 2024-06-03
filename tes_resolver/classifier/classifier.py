from abc import ABC, abstractmethod
import numpy as np

from tes_resolver.traces import Traces
import tes_resolver.config as config

class Classifier(ABC):

    def __init__(self, modeltype, **kwargs):
        self._params={
            'modeltype': modeltype,
            'time_stamp': config.time_stamp,
            'accuracy_score': np.nan
        }

    @property
    def time_stamp(self):
        return self._params['time_stamp']

    @property
    def modeltype(self):
        return self._params['modeltype']

    @property
    def accuracy_score(self):
        return self._params['accuracy_score']

    @abstractmethod
    def train(self, trainingTraces: Traces):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass

    # load from existing file.
    @abstractmethod
    def load(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, unknownTraces: Traces, **kwargs):
        pass

