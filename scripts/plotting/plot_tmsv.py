import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

from tes_resolver import Traces
from tes_resolver.classifier.inner_product import InnerProductClassifier
from utils import DFUtils, tvd, DataReader, RuquReader

'''Run ml classifier to classify all the data in a certain folder. '''
# parameters
cal_rep_rate = 100  # the rep rate to generate training

data_keywords = ['2nmPump', '112uW', '-1954_']
channels = ['Chan[1]', 'Chan[2]']

fig_titles = f'{cal_rep_rate}kHz_' + '_'.join(data_keywords)
# read data
sampling_rate = 5e4
sqReader = RuquReader(r'Data\squeezed states 2024_07_17')

# read data from the two channels
data_1 = sqReader.read_raw_data(f'{cal_rep_rate}kHz', channels[0], *data_keywords, concatenate=True, return_file_names=False)
ch1Traces = Traces(cal_rep_rate, data_1, parse_data=False, trigger_delay='automatic')

data_2 = sqReader.read_raw_data(f'{cal_rep_rate}kHz', channels[1], *data_keywords, concatenate=True, return_file_names=False)
ch2Traces = Traces(cal_rep_rate, data_2, parse_data=False, trigger_delay='automatic')

trace_objects = [ch1Traces, ch2Traces]

# inner product processing results
ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)
fig1, axs1 = plt.subplots(2,2, figsize=(8, 6), layout='constrained', sharex='row', sharey='row')
for i_ch, traces in enumerate(trace_objects):
    # predict with inner product classifier
    ipClassifier.train(traces)
    ipClassifier.predict(traces, update=True)
    overlaps = ipClassifier.calc_inner_prod(traces)

    # plot stegosaurus
    ax = axs1[0, i_ch]
    ax.hist(overlaps, bins=ipClassifier.num_bins, alpha=0.8, color='aquamarine')
    for bin_edge in ipClassifier.inner_prod_bins.values():
        ax.axvline(bin_edge, color='grey', linestyle='--', alpha=0.5)
    ax.set_title(f'{channels[i_ch]}')
    ax.set_xlabel('Inner Product')
    ax.set_ylabel('Counts')

    # plot pn distribution
    ax = axs1[1, i_ch]
    pn_labels, distrib = traces.pn_distribution(normalised=True)
    ax.bar(pn_labels, distrib, color='skyblue')
    ax.set_xlabel('Photon number label')
    ax.set_ylabel('Probability')

fig1.suptitle(fig_titles)

# plot heralded photon distribution
heralds = list(set(ch1Traces.labels))  # use ch1 to herald ch2
max_pn2 = max(np.max(ch2Traces.labels), max(heralds))  # maximum label in channel 2
fig2, axs2 = plt.subplot_mosaic(np.arange(12).reshape(2, 6), figsize=(12, 6), layout='constrained',
                                sharex=True, sharey=True)

heralded_counts = np.zeros((len(heralds), max_pn2+1), dtype=int)
for i_h, herald in enumerate(heralds):
    # select ch2 labels
    signal_indices = np.where(ch1Traces.labels == herald)[0]
    idler_labels = ch2Traces.labels[signal_indices]

    # count the number of each idler label
    for label in idler_labels:
        heralded_counts[i_h, label] += 1

    # plot the distribution
    ax = axs2[herald]
    ax.bar(np.arange(max_pn2+1), heralded_counts[i_h] / np.sum(heralded_counts[i_h]), color='skyblue')
    ax.set_title(f'{channels[0]}={herald}')
ax.set_xticks(np.arange(max_pn2+1))
fig2.suptitle(fig_titles)
fig2.supxlabel('Photon number')
fig2.supylabel(f'{channels[1]}  Probability')

# plot 3d bar plot of heralded counts (normalised)
fig3 = plt.figure(figsize=(10, 10))
ax3 = fig3.add_subplot(111, projection='3d')
x, y = np.meshgrid(np.arange(max_pn2+1), heralds)
x = x.flatten()
y = y.flatten()
z = np.zeros_like(x)
dx = dy = 0.5
dz = heralded_counts.flatten() / np.sum(heralded_counts)
ax3.bar3d(x, y, z, dx, dy, dz, shade=True)
ax3.set_xlabel(f'{channels[0]}')
ax3.set_ylabel(f'{channels[1]}')
ax3.set_zlabel('Probability')
fig3.suptitle(f'{cal_rep_rate}kHz_' + '_'.join(data_keywords))

plt.show()