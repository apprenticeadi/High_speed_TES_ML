import numpy as np
import matplotlib.pyplot as plt

from tes_resolver.classifier.inner_product import InnerProductClassifier
from tes_resolver.traces import Traces
from src.data_utils import DataReader
import tes_resolver.config as config

'''Test run the inner product classifier, compare between different data groups as well. '''

rep_rate = 100
raw_traces_range = list(range(1000))
save_classifiers = False

dataReader = DataReader('RawData')

data_groups = np.array(['raw_5', 'raw_6', 'raw_7', 'raw_8'])
fig1, axs1 = plt.subplot_mosaic(data_groups.reshape((2,2)), sharex=True, sharey=True, figsize=(12, 10), layout='constrained')
fig2, axs2 = plt.subplot_mosaic(data_groups.reshape((2,2)), figsize=(12, 10), layout='constrained')

classifiers = {}
trace_objects = {}

for data_group in data_groups:#
    ax1 = axs1[data_group]  # for plotting raw traces
    ax2 = axs2[data_group]  # for plotting histograms

    '''Read data'''
    data_raw = dataReader.read_raw_data(data_group, rep_rate)
    calTraces = Traces(rep_rate, data_raw, parse_data=True, trigger=0)

    trace_objects[data_group] = calTraces

    '''Plot data'''
    ax1.set_title(data_group)
    for i in raw_traces_range:
        ax1.plot(calTraces.data[i], alpha=0.1)
    ax1.set_xlabel('Samples')

    '''Train classifier'''
    ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)
    ipClassifier.train(calTraces)

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
    labels = ipClassifier.predict(calTraces)
    calTraces.labels = labels  # update the labels

    indices_dict, traces_dict = calTraces.bin_traces()  # dictionary of the indices and traces to their assigned photon number label
    characeristic_traces = calTraces.characteristic_traces()

    for pn in characeristic_traces.keys():
        ax1.plot(characeristic_traces[pn], color='red', alpha=1.)


'''Plot characteristic traces together'''
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
fig3, ax3 = plt.subplots(figsize=(12, 10))
for i, data_group in enumerate(trace_objects.keys()):
    calTraces = trace_objects[data_group]
    characeristic_traces = calTraces.characteristic_traces()

    for pn in characeristic_traces.keys():
        if pn == 0 :
            ax3.plot(characeristic_traces[pn], color=colors[i], alpha=0.5, label=data_group)
        else:
            ax3.plot(characeristic_traces[pn], color=colors[i], alpha=0.5,)
ax3.legend()
ax3.set_xlabel('Samples')


'''Save classifiers'''
if save_classifiers:
    for data_group in classifiers.keys():
        classifier = classifiers[data_group]
        classifier.save(rf'{data_group}_trained_{config.time_stamp}.pkl')


'''Use the classifier trained by one group and apply it to classify another group- test model persistence'''
train_data_group = 'raw_7'
test_data_group = 'raw_8'

selfClassifier = classifiers[test_data_group]
testTraces = trace_objects[test_data_group]  # result of self classification
test_pns, test_counts = testTraces.pn_distribution(normalised=False)

crossClassifier = classifiers[train_data_group]
cross_labels = crossClassifier.predict(testTraces)
crosspredictedTraces = Traces(rep_rate, testTraces.data, labels=cross_labels, parse_data=False)  # result of cross classification
cross_pns, cross_counts = crosspredictedTraces.pn_distribution(normalised=False)

'''Plot pn distributions'''
fig4, ax4 = plt.subplots(layout='constrained', figsize=(8,8))

width=0.4
bar_params = {'width': width, 'align': 'center', 'alpha':0.8}

ax = ax4
ax.bar(test_pns - width/2, test_counts, label=f'{test_data_group} trained IPClassifier', **bar_params)
ax.bar(cross_pns + width/2, cross_counts, label=f'{train_data_group} trained IPClassifier', **bar_params)
ax.set_xlabel('Photon number')
ax.set_ylabel('Counts')
ax.set_title(f'Classifying {test_data_group} 100kHz')

ax.legend()

'''Plot the inner product histograms'''
fig5, axs5 = plt.subplots(2,1, figsize=(8, 10), layout='constrained', sharey=True)

ax = axs5[0]
test_overlaps = selfClassifier.calc_inner_prod(testTraces)

ax.hist(test_overlaps, bins=selfClassifier.num_bins, alpha=0.5, label=test_data_group)
for pn_bin in selfClassifier.inner_prod_bins.values():
    ax.axvline(pn_bin, ymin=0, ymax=0.5, ls='dashed')
ax.legend()
ax.set_title(f'IPClassifier trained by {test_data_group}')

# ip classifier which is trained by a different train_data_group and then applied on test_data_group.
ax = axs5[1]

train_overlaps = crossClassifier.calc_inner_prod(trace_objects[train_data_group])
cross_overlaps = crossClassifier.calc_inner_prod(crosspredictedTraces)

ax.hist(train_overlaps, bins=crossClassifier.num_bins, alpha=0.5, label=train_data_group)
ax.hist(cross_overlaps, bins=crossClassifier.num_bins, alpha=0.5, label=test_data_group)
for pn_bin in crossClassifier.inner_prod_bins.values():
    ax.axvline(pn_bin, ymin=0, ymax=0.5, ls='dashed')
ax.set_title(f'IPClassifier trained by {train_data_group}')
ax.set_xlabel('Inner product')
ax.legend()

plt.show()





