import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import time
import os

import pandas as pd

from tes_resolver import Traces, DataChopper, config
from tes_resolver.classifier import InnerProductClassifier

from utils import DFUtils, DataReader, RuquReader

modeltype = 'IP'

sampling_rate = 5e4  # kHZ

rep_rate = 800  # kHZ
chop = False  # whether to chop the data into effective 200kHz
data_date = '2024-07-17-2010'
data_keywords = ['2nmPump', '900uW', data_date]
channels = ['Chan[1]', 'Chan[2]']
fig_titles = f'{rep_rate}kHz_' + '_'.join(data_keywords)

sqReader = RuquReader(r'Data\squeezed states 2024_07_17')
ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)

results_dir = os.path.join(config.home_dir, '..', 'Results', 'squeezed states 2024_07_17', modeltype,
                           f'{rep_rate}kHz_{data_date}_chop={chop}_{config.time_stamp}')

fig1, axs1 = plt.subplots(3, 2, figsize=(12, 7), layout='constrained', sharex='row', sharey='row')
for i_ch, channel in enumerate(channels):
    print(f'\nProcessing {channel}...')
    data = sqReader.read_raw_data(f'{rep_rate}kHz', channel, *data_keywords, concatenate=True, return_file_names=False)

    if chop:
        data = data[:, :int(data.shape[1]/2)]
        effective_rep_rate = 2*rep_rate
    else:
        effective_rep_rate = rep_rate

    traces = Traces(effective_rep_rate, data, parse_data=True, trigger_delay='automatic')

    ipClassifier.train(traces)
    ipClassifier.predict(traces, update=True)
    # ipClassifier.target_trace = traces.average_trace()
    overlaps = ipClassifier.calc_inner_prod(traces)

    # save and plot first 1000 traces
    data_sample = traces.data[:1000]
    np.savetxt(DFUtils.create_filename(results_dir + rf'\{channel}_first_1000.txt'), data_sample)
    ax = axs1[0, i_ch]
    for i in range(1000):
        ax.plot(data_sample[i], alpha=0.1)
    ax.set_title(f'{channels[i_ch]}')
    ax.set_xlabel('Sample')

    # save the pn labels
    raw_labels = traces.labels
    np.savetxt(DFUtils.create_filename(results_dir + rf'\{channel}_pn_labels.txt'), raw_labels)

    # plot stegosaurus and pn distribution
    ax = axs1[1, i_ch]
    heights, bin_edges, _ = ax.hist(overlaps, bins=ipClassifier.num_bins, alpha=0.8, color='aquamarine')
    heights = np.append(heights, 0)

    for bin_edge in ipClassifier.inner_prod_bins.values():
        ax.axvline(bin_edge, color='grey', linestyle='--', alpha=0.5)
    ax.set_title(f'{channels[i_ch]}')
    ax.set_xlabel('Inner Product')
    ax.set_ylabel('Counts')

    ax = axs1[2, i_ch]
    pn_labels, distrib = traces.pn_distribution(normalised=True)
    ax.bar(pn_labels, distrib, color='skyblue')
    ax.set_xlabel('Photon number label')
    ax.set_ylabel('Probability')

    # save histogram and pn distribution
    np.savetxt(results_dir + rf'\{channel}_stegosaurus.txt', np.stack((bin_edges, heights)))
    np.savetxt(results_dir + rf'\{channel}_pn_distribution.txt', np.stack((pn_labels, distrib)))

fig1.suptitle(fig_titles)
fig1.savefig(DFUtils.create_filename(results_dir + rf'\{modeltype}_stegosaurus.pdf'))