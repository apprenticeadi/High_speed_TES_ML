import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from src.data_reader import DataReader
from src.utils import DFUtils
from tes_resolver.traces import Traces
from tes_resolver.data_chopper import DataChopper

dataReader = DataReader(r'\Data\RawData_2023_07')

data_group = 'raw_8'

# save_dir = r'../../Plots/trace_plots'

data100_raw = dataReader.read_raw_data(data_group, rep_rate=100)
refTraces = Traces(100, data100_raw, parse_data=True, trigger_delay=0)
data100 = refTraces.data

high_freq = 800
period = int(5e4  / high_freq)
data_high_raw = dataReader.read_raw_data(data_group, rep_rate=high_freq)
trigger_delay = DataChopper.find_trigger(data_high_raw, samples_per_trace= period)
highTraces = Traces(high_freq, data_high_raw, parse_data=True, trigger_delay=trigger_delay)
data_high = highTraces.data

'''Plot trace trains'''
plt.figure('100kHz trace train')
plt.plot( data100[:10].flatten())
plt.xlabel('Samples')
plt.ylabel(r'$uV$')
plt.xlim(0, 2*500)
plt.ylim(-1000, 16500)
# plt.savefig(DFUtils.create_filename(save_dir + rf'\{data_group}_100kHz_trace_train.pdf'))

plt.figure(f'{high_freq}kHz trace train')
plt.plot(data_high[: 5000 // period + 1].flatten())
plt.xlabel('Samples')
plt.ylabel(r'$uV$')
plt.xlim(0, 2*500)
plt.ylim(-1000, 16500)
# plt.savefig(DFUtils.create_filename(save_dir + rf'\{data_group}_{high_freq}kHz_trace_train.pdf'))

'''Traces'''
to_show = 1000
data100_to_plot = data100[:2*to_show].reshape((to_show, 2*500))
data_high_to_plot = data_high[:2*to_show].reshape((to_show, 2*period))

fontsize = 14
# plot together
fig, axs = plt.subplot_mosaic([['(a) 100kHz'], [f'(b) {high_freq}kHz']], figsize=(8,6), layout='constrained', sharey=True)
# plt.subplots_adjust(hspace=0.5, wspace=0.)

for label, ax in axs.items():
    ax.set_title(label, fontfamily='serif', loc='left', fontsize=fontsize+2)

ax1 = axs['(a) 100kHz']
for i in range(to_show):
    ax1.plot(np.arange(data100_to_plot.shape[1])*20/1000, data100_to_plot[i],alpha=0.05)
ax1.set_xlabel(r'$\mu s$', fontsize=fontsize)
# ax1.set_ylabel(r'$\mu V$', fontsize=fontsize)  # this unit is probably not correct. need to divide alazar card range with resolution (should be 14bits)
ax1.set_ylim(-1000, 20000)
ax1.set_xlim(0, 20)
ax1.set_xticks(np.arange(0, 20, 5))
ax1.tick_params(labelsize=fontsize-2)

ax2 = axs[f'(b) {high_freq}kHz']
for i in range(to_show):
    ax2.plot(np.arange(data_high_to_plot.shape[1])*20/1000, data_high_to_plot[i], alpha=0.05)
ax2.set_xlabel(r'$\mu s$', fontsize=fontsize)
ax2.set_ylim(-1000, 20000)
# ax2.set_yticks([])
ax2.set_xlim(0, 2*period*20/1000)
ax2.tick_params(labelsize=fontsize-2)

plt.show()

# fig.savefig(DFUtils.create_filename(save_dir + rf'\{data_group}_100_vs_{high_freq}kHz_just_traces.pdf'))