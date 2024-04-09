import numpy as np
import matplotlib.pyplot as plt
import time

from tes_resolver.classifier.inner_product import InnerProductClassifier
from tes_resolver.classifier.tabular import TabularClassifier
from tes_resolver.traces import Traces
import tes_resolver.config as config

from src.data_utils import DataReader

'''Train RF and inner product classifiers with 100kHz data from one group and test it on another datagroup'''

dataReader = DataReader('RawData')
rep_rate = 100

'''Pick two data groups'''
train_data_group = 'raw_7'
train_data = dataReader.read_raw_data(train_data_group, rep_rate)
trainTraces = Traces(rep_rate, train_data, parse_data=True, trigger=0)

test_data_group = 'raw_8'
test_data = dataReader.read_raw_data(test_data_group, rep_rate)
testTraces = Traces(rep_rate, test_data, parse_data=True, trigger=0)

'''Classify training data with inner product method first.'''
ipClassifier = InnerProductClassifier()
ipClassifier.train(trainTraces)
train_labels = ipClassifier.predict(trainTraces)
trainTraces.labels = train_labels

'''Train the RF classifier with training data'''
rfClassifier = TabularClassifier('RF', test_size=0.1)
print('Training RF...')
t1 = time.time()
rfClassifier.train(trainTraces)
t2 = time.time()
print(f'RF training finished after {t2-t1}s')


'''Test the RF classifier on the test data'''
t1 = time.time()
rf_test_labels = rfClassifier.predict(testTraces)
t2 = time.time()
print(f'RF prediction finished after {t2-t1}s')

'''The groundtruth labels by training ip classifier on the test data itself'''
groundtruthClassifier = InnerProductClassifier()
groundtruthClassifier.train(testTraces)
gt_test_labels = groundtruthClassifier.predict(testTraces)
testTraces.labels = gt_test_labels

gt_labels, gt_counts = testTraces.pn_distribution(normalised=False)

'''Compare the results'''
rf_predictedTraces = Traces(rep_rate, testTraces.data, labels=rf_test_labels, parse_data=False)
rf_labels, rf_counts = rf_predictedTraces.pn_distribution(normalised=False)


'''Plot and compare the distributions from the three classifiers'''
fig, ax = plt.subplots(layout='constrained', figsize=(8,8))

width=0.4
bar_params = {'width': width, 'align': 'center', 'alpha':0.8}

ax.bar(gt_labels - width/2, gt_counts, label=f'{test_data_group} trained IPClassifier', **bar_params)
ax.bar(rf_labels + width/2, rf_counts, label=f'{train_data_group} trained RFClassifier', **bar_params)
ax.set_xlabel('Photon number')
ax.set_ylabel('Counts')
ax.set_title(f'Classifying {test_data_group} 100kHz')

ax.legend()

plt.show()


'''Save classifier'''
rfClassifier.save(filename=f'RF_{rep_rate}kHz_trained_by_{train_data_group}_{config.time_stamp}')

'''Load classifier again'''
loadClassifier = TabularClassifier()
loadClassifier.load(f'RF_{rep_rate}kHz_trained_by_{train_data_group}_{config.time_stamp}')
