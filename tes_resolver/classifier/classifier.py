from abc import ABC, abstractmethod

from tes_resolver.traces import Traces

class Classifier(ABC):

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
    def predict(self, unknownTraces: Traces):
        pass

