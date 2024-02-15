import warnings

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import xgboost
from src.utils import DataUtils
from src.traces import Traces
import numpy as np
from scipy.signal import welch


class ML:
    '''
    class which contains the tabular models
    '''
    def __init__(self, dataset, labels, modeltype = 'RF', n_estimators = 400, max_depth = 5):

        self.dataset = dataset
        self.labels = labels
        self.modeltype = modeltype
        # default test size is 25%
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dataset, self.labels)

        if self.modeltype == 'RF':
            self.classifier = RandomForestClassifier(n_estimators=n_estimators)

        elif self.modeltype == 'SVM':
            self.classifier = SVC()

        elif self.modeltype == 'BDT':
            self.classifier = xgboost.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators)

        elif self.modeltype =='KNN':
            self.classifier = KNeighborsClassifier()

        else:
            raise Exception('modeltype must be "RF", "SVM", "BDT" or "KNN" (Random forest, support vector machines, boosted decision tree and K-nearest neighbors)')

    def makemodel(self):
        # x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.labels)
        self.classifier.fit(self.x_train, self.y_train)

    def accuracy_score(self):
        predictions = self.classifier.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy

    def predict(self, targets):
        predictions = self.classifier.predict(targets)
        return predictions

    def classification_report(self):
        predictions = self.classifier.predict(self.x_test)
        report = classification_report(self.y_test, predictions)
        return report

    def confusion_matrix(self):
        predictions = self.classifier.predict(self.x_test)
        matrix = confusion_matrix(self.y_test, predictions)
        return matrix

    def ROC_curve(self):
        predictions = self.classifier.predict(self.x_test)
        return roc_curve(self.y_test, predictions)


def return_artifical_data(frequency, multiplier, power):
    '''
    function which returns an artificial dataset with its corresponding PN label
    '''
    warnings.warn('This method should be deprecated. Use Traces.generate_training_data() instead')

    num_bins = 1000
    data_100 = DataUtils.read_raw_data_new(100, power)
    calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)

    labels = calibrationTraces.return_pn_labels()

    # Filter out traces that were not labelled due to choice of multiplier
    filtered_ind = np.where(labels == -1)[0]
    filtered_traces = np.delete(data_100, filtered_ind, axis=0)
    filtered_label = np.delete(labels, filtered_ind)
    filtered_data = Traces(100, filtered_traces)

    data_high = filtered_data.generate_high_freq_data(frequency)

    return data_high, filtered_label[2:]


def extract_features(x):
    '''
    function which extracts features from a trace x and returns a feature vector
    (INCOMPLETE) add more relevant features
    '''
    peaks, props = find_peaks(x)
    peak_heights = x[peaks]

    if len(peaks)==0:
        peak_loc, max_peak = np.argmax(x), max(x)
    if len(peaks)>0:
        peak_loc, max_peak = peaks[np.argmax(peak_heights)], max(peak_heights)

    average = np.mean(x)
    std = np.std(x)
    y = np.argwhere(x ==max_peak/2)

    if len(y) ==0:
        rise_time = peak_loc/2
    if len(y)>0:
        rise_time = np.abs(y[0][0]-peak_loc)
    energy = np.sum(x**2)
    frequencies, psd = welch(x)
    if len(frequencies)==0:
        freq = 0
    if len(frequencies) >0:
        freq = frequencies[np.argmax(psd)]
    crest = max_peak/np.sqrt(np.mean(x**2))
    var = np.var(x)
    kurt = (np.sum((x - average)**4)/(var ** 2)) - 3
    area = trapezoid(x)
    return [peak_loc,average,  std, energy, freq, max_peak, rise_time,crest, kurt, area]


def find_offset(frequency, power):
    '''
    function which find the offset by average trace peak height and returns the difference
    '''
    actual_data = DataUtils.read_high_freq_data(frequency, power=power, new=True)
    data_high, labels = return_artifical_data(frequency, 1.6,power)
    art_trace, actual_trace = Traces(frequency, data_high, 2), Traces(frequency, actual_data, 2)
    av1, a, b = actual_trace.average_trace()
    av2, c, d = art_trace.average_trace()
    shift = np.max(av1) - np.max(av2)
    return shift