import numpy as np
import xgboost
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from tes_resolver.classifier.classifier import Classifier
from tes_resolver.traces import Traces
import tes_resolver.config as config

class TabularClassifier(Classifier):

    def __init__(self, modeltype='RF', test_size=0.25, **classifier_kwargs):
        """A wrapper class for tabular classifiers. """

        super().__init__(modeltype)

        self.test_size = test_size
        self.default_dir = os.path.join(config.home_dir, 'saved_classifiers', f'{modeltype}Classifier')

        self._params.update( {
            # 'modeltype': modeltype,
            'accuracy_score': 0. ,
            # 'time_stamp': config.time_stamp,
        })

        if self.modeltype == 'RF':
            c_params = {'n_estimators': 400}
            c_params.update(classifier_kwargs)
            self.classifier = RandomForestClassifier(**c_params)

        elif self.modeltype == 'SVM':
            self.classifier = SVC(**classifier_kwargs)

        elif self.modeltype == 'BDT':
            c_params = {'max_depth': 5, 'n_estimators': 400}
            c_params.update(classifier_kwargs)
            self.classifier = xgboost.XGBClassifier(**c_params)

        elif self.modeltype == 'KNN':
            self.classifier = KNeighborsClassifier(**classifier_kwargs)

        else:
            raise ValueError('modeltype must be "RF", "SVM", "BDT" or "KNN" (Random forest, support vector machines, '
                             'boosted decision tree and K-nearest neighbors)')

    # @property
    # def time_stamp(self):
    #     return self._params['time_stamp']

    # @property
    # def accuracy_score(self):
    #     return self._params['accuracy_score']

    # @property
    # def modeltype(self):
    #     return self._params['modeltype']

    def train(self, trainingTraces: Traces):
        """Train model with given training Traces and update accuracy score with test data."""
        data = trainingTraces.data
        labels = trainingTraces.labels

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=self.test_size)
        self.classifier.fit(x_train, y_train)

        predictions_test = self.classifier.predict(x_test)
        self._params['accuracy_score'] = accuracy_score(y_test, predictions_test)

    def predict(self, unknownTraces: Traces, update=False):
        """Wrapper method around the predict method of the classifier itself. """
        labels = self.classifier.predict(unknownTraces.data)

        if update:
            unknownTraces.labels = labels

        return labels

    def save(self, filename=None, filedir=None):
        if filename is None:
            filename = fr'{self.modeltype}_{self.time_stamp}.pkl'
        elif filename[-4:] != r'.pkl':
            filename = filename + r'.pkl'

        if filedir is None:
            filedir = self.default_dir
        os.makedirs(filedir, exist_ok=True)

        # save classifier
        fullfilename = os.path.join(filedir, filename)
        with open(fullfilename, 'wb') as output_file:
            pickle.dump(self.classifier, output_file)

        # save params
        paramsfilename = fullfilename[:-4] + '_params' + fullfilename[-4:]
        with open(paramsfilename, 'wb') as output_file:
            pickle.dump(self._params, output_file)

    def load(self, filename, filedir=None):
        if filename[-4:] != r'.pkl':
            filename = filename + r'.pkl'

        if filedir is None:
            filedir = self.default_dir

        fullfilename = os.path.join(filedir, filename)

        with open(fullfilename, 'rb') as input_file:
            self.classifier = pickle.load(input_file)

        paramsfilename = fullfilename[:-4] + '_params' + fullfilename[-4:]
        if os.path.isfile(paramsfilename):
            with open(paramsfilename, 'rb') as input_file:
                params = pickle.load(input_file)
            self._params.update(params)