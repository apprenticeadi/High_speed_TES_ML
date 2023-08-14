from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from src.utils import DataUtils
from src.composite_funcs import return_comp_traces
from src.traces import Traces
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
#from sktime.classification.kernel_based import RocketClassifier
#from sktime.transformations.panel.catch22 import Catch22
#from sktime.pipeline import make_pipeline

class ML:

    def __init__(self, dataset, labels, modeltype = 'RF'):

        self.dataset = dataset
        self.labels = labels
        self.modeltype = modeltype

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dataset, self.labels)

        if self.modeltype == 'RF':
            self.classifier = RandomForestClassifier()

        elif self.modeltype == 'SVM':
            self.classifier = SVC()

        elif self.modeltype == 'BDT':
            self.dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
            self.dtest = xgb.DMatrix(self.x_test, label=self.y_test)

        # elif self.modeltype == 'RKT':
        #     self.classifier = RidgeClassifierCV()
        #
        # elif self.modeltype == 'C22':
        #     catch22 = Catch22()
        #     randf = RandomForestClassifier()
        #     pipe = make_pipeline(catch22,randf)
        #     self.classifier = pipe

        elif self.modeltype =='KNN':
            self.classifier = KNeighborsClassifier()

        else:
            raise Exception('modeltype must be "RF", "SVM" or  "BDT" (Random forest, support vector machines or boosted decision tree)')

    def makemodel(self, num_rounds = 10, num_class = 10):

        if self.modeltype == 'RF':
            print('Building Random Forest')
            x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.labels)
            self.classifier.fit(x_train, y_train)
        if self.modeltype == 'SVM':
            print('Building support vector machines')
            x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.labels)
            self.classifier.fit(x_train, y_train)
        if self.modeltype == 'BDT':
            print('Building boosted decision tree')
            params = {
                'max_depth': 5,
                'eta': 0.1,
                'num_class': num_class,
                'objective': 'multi:softmax'
            }
            self.classifier = xgb.train(params, self.dtrain, num_rounds)
        if self.modeltype =='RKT':
            print('Building Rocket')
            x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.labels)
            rocket = RocketClassifier()
            self.classifier.fit(x_train,y_train)
        if self.modeltype == 'C22':
            print('Building catch22')
            x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.labels)
            tqdm(self.classifier.fit(x_train,y_train))
        if self.modeltype =='KNN':
            print('Building K-Nearest neighbors')
            x_train, x_test, y_train, y_test = train_test_split(self.dataset, self.labels)
            self.classifier.fit(x_train, y_train)


    def accuracy_score(self):
        if self.modeltype == 'BDT':
            predictions = self.classifier.predict(self.dtest)
            accuracy = accuracy_score(self.y_test, predictions)
            return accuracy
        else:
            predictions = self.classifier.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, predictions)
            return accuracy


    def predict(self, targets):
        if self.modeltype == 'BDT':
            d_matrix = xgb.DMatrix(targets)
            predictions = self.classifier.predict(d_matrix)
            return predictions.astype(int)
        else:
            predictions = self.classifier.predict(targets)
            return predictions

    def classifacation_report(self):
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

def find_accuracies(calibrationTraces):
    freq_values =[ 500,600,700,800,900]
    model_types = ['RF', 'SVM', 'BDT']
    rf_scores = []
    svm_scores = []
    bdt_scores = []
    scores = (rf_scores, svm_scores, bdt_scores)
    for x in tqdm(freq_values):
        frequency = x
        data_high = DataUtils.read_high_freq_data(frequency)
        targetTraces = Traces(frequency=frequency, data=data_high, multiplier=1.2, num_bins=1000)
        freq_str = targetTraces.freq_str
        #
        pn_combs, comp_traces = return_comp_traces(calibrationTraces, targetTraces)
        '''
        create a labelled dataset for training/ testing, labelled_comp_traces is a list of all traces with photon number as index
        '''
        labelled_comp_traces = []
        labelled_pn_combs = []

        for i in range(10):
            indices = np.arange(i, 3000, 10)
            new_array = comp_traces[indices]
            new_arry1 = pn_combs[indices]
            labelled_comp_traces.append(new_array)
            labelled_pn_combs.append((new_arry1))
        '''
        creating dataset
        '''
        dataset = np.concatenate((labelled_comp_traces[0], labelled_comp_traces[1], labelled_comp_traces[2],
                                  labelled_comp_traces[3], labelled_comp_traces[4], labelled_comp_traces[5],
                                  labelled_comp_traces[6],
                                  labelled_comp_traces[7], labelled_comp_traces[8], labelled_comp_traces[9]))

        num = len(labelled_comp_traces[0])

        labels = np.array([0] * num + [1] * num + [2] * num + [3] * num + [4] * num + [5] * num +
                          [6] * num + [7] * num + [8] * num + [9] * num)
        for j in range(3):
            model = ML(dataset, labels, modeltype=model_types[j])
            model.makemodel()
            scr = model.accuracy_score()
            scores[j].append(scr)
    return freq_values, scores, model_types

def visualise_trace(labelled_comp_traces, predictions,data_high, num):
    num1 = predictions[num]
    if num1 == 0:
        print('identified trace as 0')
    a = str(num1)
    b = str(num1 + 1)
    c = str(num1 - 1)
    l = 0.2

    for i in range(1, len(labelled_comp_traces[num1])):
        plt.plot(labelled_comp_traces[num1][i], color='r', linestyle='dashed', linewidth=0.7)

    for i in range(1, len(labelled_comp_traces[num1])):
        plt.plot(labelled_comp_traces[num1 + 1][i], color='g', linestyle='dashed', linewidth=l)
    plt.plot(labelled_comp_traces[num1 + 1][0], color='g', linestyle='dashed', linewidth=0.5,
             label='composite traces for ' + b)
    for i in range(1, len(labelled_comp_traces[num1])):
        plt.plot(labelled_comp_traces[num1 - 1][i], color='lawngreen', linestyle='dashed', linewidth=l)
    plt.plot(labelled_comp_traces[num1 - 1][i], color='y', linestyle='dashed', linewidth=0.5,
             label='composite traces for ' + c)
    plt.plot(labelled_comp_traces[num1][0], color='r', linestyle='dashed', linewidth=0.7,
             label='composite traces for ' + a)
    plt.plot(data_high[num], color='k', label='target curve')
    plt.legend()
    plt.title('composite curves for ' + a)
    plt.show()

def return_artifical_data(frequency, multiplier, power):
    num_bins = 1000
    data_100 = DataUtils.read_raw_data_new(100, power)
    calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
    labels = calibrationTraces.return_labelled_traces()
    filtered_ind = np.where(labels == -1)[0]
    filtered_traces = np.delete(data_100, filtered_ind, axis=0)
    filtered_label = np.delete(labels, filtered_ind)
    filtered_data = Traces(100, filtered_traces)
    data_high = filtered_data.overlap_to_high_freq(frequency)
    return data_high, filtered_label



