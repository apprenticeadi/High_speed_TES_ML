from abc import ABC, abstractmethod

from tes_resolver.traces import Traces

class Classifier(ABC):

    @abstractmethod
    def train(self, calTraces: Traces):
        pass

    @abstractmethod
    def save(self):
        pass

    # load from existing file.
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self):
        pass

