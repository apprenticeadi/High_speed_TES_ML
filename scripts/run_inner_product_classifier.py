import numpy as np
import matplotlib.pyplot as plt

from tes_resolver.classifier.inner_product import InnerProductClassifier
from tes_resolver.traces import Traces
from src.data_utils import DataReader

'''Test run the inner product classifier, compare between different data groups as well. '''

rep_rate = 100

raw_traces_range = list(range(1000))

data_groups = np.array(['raw_5', 'raw_6', 'raw_7', 'raw_8'])
fig1, axs1 = plt.subplot_mosaic(data_groups.reshape((2,2)), sharex=True, sharey=True, figsize=(12, 10), layout='constrained')
fig2, axs2 = plt.subplot_mosaic(data_groups.reshape((2,2)), figsize=(12, 10), layout='constrained')

classifiers = {}
trace_objects = {}

# Todo: try choosing one single target Trace to bin the overlaps from all 4 data groups!
# Maybe try pooling together the data groups in random order?

for data_group in data_groups:#
    ax1 = axs1[data_group]  # for plotting raw traces
    ax2 = axs2[data_group]  # for plotting histograms

    '''Read data'''
    dataReader = DataReader(data_group)
    data_raw = dataReader.read_raw_data(rep_rate)
    calTraces = Traces(rep_rate, data_raw, parse_data=True)

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

plt.show()