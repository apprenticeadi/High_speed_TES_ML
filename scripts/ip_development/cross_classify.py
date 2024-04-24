import numpy as np
import matplotlib.pyplot as plt
import time

from tes_resolver.classifier.inner_product import InnerProductClassifier
from tes_resolver.traces import Traces
from src.data_reader import DataReader
import tes_resolver.config as config

'''Compare between different data groups, and test out cross-classification, which means training the ip classifier with
 one group of data with higher photon number, and use it to predict another group of data with lower photon number. '''

rep_rate = 100
# raw_traces_range = list(range(100))
save_classifiers = False

dataReader = DataReader('Tomography_data_2024_04')

powers = np.arange(12)
data_groups = np.array([f'power_{p}' for p in powers])
# fig1, axs1 = plt.subplot_mosaic(data_groups.reshape((2,6)), sharex=True, sharey=True, figsize=(18, 6), layout='constrained')
fig2, axs2 = plt.subplot_mosaic(data_groups.reshape((2,6)), figsize=(18, 6), layout='constrained')

classifiers = {}
trace_objects = {}

for data_group in data_groups:#
    # ax1 = axs1[data_group]  # for plotting raw traces
    ax2 = axs2[data_group]  # for plotting histograms

    '''Read data'''
    data_raw = dataReader.read_raw_data(data_group, rep_rate)
    calTraces = Traces(rep_rate, data_raw, parse_data=True, trigger=0)

    trace_objects[data_group] = calTraces

    # '''Plot data'''
    # ax1.set_title(data_group)
    # for i in raw_traces_range:
    #     ax1.plot(calTraces.data[i], alpha=0.1)
    # ax1.set_xlabel('Samples')
    # print(f'Raw data for {data_group} plotted')

    '''Train classifier'''
    ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)
    t1 = time.time()
    ipClassifier.train(calTraces)
    t2 = time.time()
    print(f'Classifier trained on {data_group} after {t2-t1}s')

    classifiers[data_group] = ipClassifier

    '''Plot the inner products'''
    overlaps = ipClassifier.calc_inner_prod(calTraces)
    ax2.set_title(data_group)
    ax2.hist(overlaps, bins=ipClassifier.num_bins, color='aquamarine')

    # mark the bins identified by the classifier
    ip_bins = ipClassifier.inner_prod_bins
    for pn in ip_bins.keys():
        ax2.axvline(ip_bins[pn], ymin=0, ymax=0.25, ls='dashed')

    '''Classify the traces '''
    t1 = time.time()
    labels = ipClassifier.predict(calTraces)
    t2 = time.time()
    print(f'Classifier predict {data_group} after {t2-t1}s')
    calTraces.labels = labels  # update the labels

    indices_dict, traces_dict = calTraces.bin_traces()  # dictionary of the indices and traces to their assigned photon number label
    characeristic_traces = calTraces.characteristic_traces()

    # for pn in characeristic_traces.keys():
    #     ax1.plot(characeristic_traces[pn], color='red', alpha=1.)


'''Plot characteristic traces together'''
fig3, ax3 = plt.subplots(figsize=(12, 10))
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

to_plot = list(trace_objects.keys())[1::2]
for i, data_group in enumerate(to_plot):
    calTraces = trace_objects[data_group]
    characeristic_traces = calTraces.characteristic_traces()

    for pn in characeristic_traces.keys():
        if pn == 0 :
            ax3.plot(characeristic_traces[pn], color=colors[i], alpha=1, label=data_group)
        else:
            ax3.plot(characeristic_traces[pn], color=colors[i], alpha=1)
ax3.legend()
ax3.set_xlabel('Samples')


'''Save classifiers'''
if save_classifiers:
    for data_group in classifiers.keys():
        classifier = classifiers[data_group]
        classifier.save(rf'{data_group}_trained_{config.time_stamp}.pkl')


'''Use the classifier trained by one group and apply it to classify another group- test model persistence'''
train_data_group = data_groups[-1]

crossClassifier = classifiers[train_data_group]
'''Plot the inner product histograms'''
fig5, ax5 = plt.subplots(1,1, figsize=(15, 8), layout='constrained', sharey=True)

for test_data_group in data_groups:
    testTraces = trace_objects[test_data_group]
    test_overlaps = crossClassifier.calc_inner_prod(testTraces)

    ax5.hist(test_overlaps, bins=crossClassifier.num_bins, alpha=0.1, label=test_data_group)

for pn_bin in crossClassifier.inner_prod_bins.values():
    ax5.axvline(pn_bin, ymin=0, ymax=0.5, ls='dashed')

ax5.set_title(f'IPClassifier trained by {train_data_group}')
ax5.set_xlabel('Inner product')
ax5.legend()

plt.show()





