import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import string

from src.data_reader import DataReader
from src.utils import DFUtils
from tes_resolver.traces import Traces
from tes_resolver.data_chopper import DataChopper
from tes_resolver.classifier import InnerProductClassifier

'''Which data to use'''
dataReader = DataReader(r'\Data\Tomography_data_2024_04')
data_group = 'power_6'
save_dir = r'../../Plots/Tomography_data_2024_04/trace_plots'

rep_rates = [100, 500, 800]

'''Parameters'''
sampling_rate = 5e4
card_range_pm = 1  # card range is +/- 1V
voltage_precision = 2 * card_range_pm / (np.power(2, 14)-1)  # unit V

num_traces = 1000
fontsize = 14
alphabet = list(string.ascii_lowercase)

fig, axs = plt.subplots(2, len(rep_rates), sharey='row', layout='constrained', figsize=(16, 6))
traces_dict = {}
for i_rep, rep_rate in enumerate(rep_rates):
    '''Read traces'''
    data_raw = dataReader.read_raw_data(data_group, rep_rate=rep_rate)
    if rep_rate <= 300:
        trigger_delay = 0
    else:
        trigger_delay = DataChopper.find_trigger(data_raw, samples_per_trace=int(sampling_rate / rep_rate))
    curTraces = Traces(rep_rate, data_raw, parse_data=True, trigger_delay=trigger_delay)
    traces_dict[rep_rate] = curTraces

    '''Label traces'''
    ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)
    ipClassifier.train(curTraces)
    ipClassifier.predict(curTraces, update=True)

    '''Save the data as txt'''
    data_parsed = curTraces.data
    data_to_plot = data_parsed[:2*num_traces].reshape((num_traces, 2 * curTraces.period))
    data_to_plot = data_to_plot /  4 * voltage_precision * 1000
    np.savetxt(DFUtils.create_filename(save_dir + rf'\{data_group}_{rep_rate}kHz_first_{num_traces}traces.txt'),
               data_to_plot, delimiter=',')

    '''Plotting'''
    ax = axs[0, i_rep]
    ax.set_title(f'({alphabet[i_rep]}) {rep_rate}kHz', fontfamily='serif', loc='left', fontsize=fontsize + 2)
    for i in range(num_traces):
        ax.plot(np.arange(data_to_plot.shape[1]) / sampling_rate * 1000, data_to_plot[i],
                alpha=0.05)

    if i_rep == 0:
        ax.set_ylabel(r'$mV$', fontsize=fontsize)
    ax.set_xlabel(r'$\mu s$', fontsize=fontsize)
    ax.set_xlim(0, data_to_plot.shape[1] / sampling_rate * 1000)
    ax.tick_params(labelsize=fontsize - 2)

    '''Plot Stegosaurus'''
    ax = axs[1, i_rep]
    # ax.set_title(f'({alphabet[i_rep + len(rep_rates)]}) {rep_rate}kHz', fontfamily='serif', loc='left', fontsize=fontsize + 2)

    # plot histogram
    overlaps = ipClassifier.calc_inner_prod(curTraces)
    inner_prod_bins = ipClassifier.inner_prod_bins
    hist_object = ax.hist(overlaps, bins=ipClassifier.num_bins, color='darkgrey')

    # save data
    heights = hist_object[0]
    bin_edges = hist_object[1]
    np.savetxt(save_dir + rf'\{data_group}_{rep_rate}kHz_stegosaurus_heights.txt', heights, delimiter=',')
    np.savetxt(save_dir + rf'\{data_group}_{rep_rate}kHz_stegosaurus_bin_edges.txt', bin_edges, delimiter=',')

    # label peaks
    if rep_rate == 100:
        for pn in inner_prod_bins.keys():
            overlap_upper_lim = inner_prod_bins[pn]
            if hist_object[1][-1] == overlap_upper_lim:
                upper_bin = -1
            else:
                upper_bin = np.argmax(hist_object[1] > overlap_upper_lim)
            if pn == 0:
                lower_bin = 0

                # central_overlap = np.mean([np.min(overlaps), inner_prod_bins[pn]])
            else:
                overlap_lower_lim = inner_prod_bins[pn-1]
                lower_bin = np.argmax(hist_object[1] > overlap_lower_lim)

                # central_overlap = np.mean([inner_prod_bins[pn-1], inner_prod_bins[pn]])
            # position = np.argmax(hist_object[1] > central_overlap)
            position = np.argmax(hist_object[0][lower_bin:upper_bin]) + lower_bin  # position of the highest peak in the pn bin
            ax.text(hist_object[1][position], hist_object[0][position]*1.1, pn)

    ax.set_xlabel('Inner product', fontsize=fontsize)
    if i_rep == 0:
        ax.set_ylabel('Counts', fontsize=fontsize)
    ax.set_xlim(left=0)
    ax.tick_params(labelsize=fontsize - 2)

plt.show()

ax.set_yscale('symlog', linthresh=1)
ax.set_ylim(0, 5000)
fig.savefig(DFUtils.create_filename(save_dir + rf'\{data_group}_trace_stegosaurus_log.pdf'))

#
# data100_raw = dataReader.read_raw_data(data_group, rep_rate=100)
# refTraces = Traces(100, data100_raw, parse_data=True, trigger_delay=0)
# data100 = refTraces.data
#
# high_freq = 800
# period = int(5e4  / high_freq)
# data_high_raw = dataReader.read_raw_data(data_group, rep_rate=high_freq)
# trigger_delay = DataChopper.find_trigger(data_high_raw, samples_per_trace= period)
# highTraces = Traces(high_freq, data_high_raw, parse_data=True, trigger_delay=trigger_delay)
# data_high = highTraces.data
#
# '''Traces'''
# to_show = 1000
# data100_to_plot = data100[:2*to_show].reshape((to_show, 2*500))
# data_high_to_plot = data_high[:2*to_show].reshape((to_show, 2*period))
#
#
# # plot together
# fig, axs = plt.subplot_mosaic([['(a) 100kHz'], [f'(b) {high_freq}kHz']], figsize=(8,6), layout='constrained', sharey=True)
# # plt.subplots_adjust(hspace=0.5, wspace=0.)
#
# for label, ax in axs.items():
#     ax.set_title(label, fontfamily='serif', loc='left', fontsize=fontsize+2)
#
# ax1 = axs['(a) 100kHz']
# for i in range(to_show):
#     ax1.plot(np.arange(data100_to_plot.shape[1])*20/1000, data100_to_plot[i],alpha=0.05)
# ax1.set_xlabel(r'$\mu s$', fontsize=fontsize)
# # ax1.set_ylabel(r'$\mu V$', fontsize=fontsize)  # this unit is probably not correct. need to divide alazar card range with resolution (should be 14bits)
# ax1.set_ylim(-1000, 20000)
# ax1.set_xlim(0, 20)
# ax1.set_xticks(np.arange(0, 20, 5))
# ax1.tick_params(labelsize=fontsize-2)
#
# ax2 = axs[f'(b) {high_freq}kHz']
# for i in range(to_show):
#     ax2.plot(np.arange(data_high_to_plot.shape[1])*20/1000, data_high_to_plot[i], alpha=0.05)
# ax2.set_xlabel(r'$\mu s$', fontsize=fontsize)
# ax2.set_ylim(-1000, 20000)
# # ax2.set_yticks([])
# ax2.set_xlim(0, 2*period*20/1000)
# ax2.tick_params(labelsize=fontsize-2)

