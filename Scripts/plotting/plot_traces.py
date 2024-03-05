import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from src.data_utils import DataParser
from src.utils import DFUtils
from src.traces import Traces

sub_dir = 'raw_8'
dataParser = DataParser(sub_dir)

save_dir = r'../../Plots/trace_plots'

data100 = dataParser.parse_data(100, triggered=False)
refTraces = Traces(100, data100, multiplier=1., num_bins=1000)

high_freq = 700
period = int(5e4  / high_freq)
data_high = dataParser.parse_data(high_freq, triggered=True)
highTraces = Traces(high_freq, data_high, multiplier=1., num_bins=1000)

# '''Plot trace trains'''
# plt.figure('100kHz trace train')
# plt.plot( data100[:10].flatten())
# plt.xlabel('Samples')
# plt.ylabel(r'$uV$')
# plt.xlim(0, 2*500)
# plt.ylim(-1000, 16500)
# plt.savefig(DFUtils.create_filename(save_dir + rf'\{sub_dir}_100kHz_trace_train.pdf'))
#
# plt.figure(f'{high_freq}kHz trace train')
# plt.plot(data_high[: 5000 // period + 1].flatten())
# plt.xlabel('Samples')
# plt.ylabel(r'$uV$')
# plt.xlim(0, 2*500)
# plt.ylim(-1000, 16500)
# plt.savefig(DFUtils.create_filename(save_dir + rf'\{sub_dir}_{high_freq}kHz_trace_train.pdf'))
#
# '''Plot inner product histogram'''
# plt.xlabel('Inner product')
# plt.ylabel('Counts')
# plt.ylim(0, 300)
# plt.savefig(DFUtils.create_filename(save_dir + rf'\{sub_dir}_100kHz_stegosaurus.pdf'))
#
# highTraces.raw_histogram(plot=True, fig_name=f'{high_freq}kHz raw stegosaurus')
# plt.xlabel('Inner product')
# plt.ylabel('Counts')
# plt.ylim(bottom=0)
# plt.savefig(DFUtils.create_filename(save_dir + rf'\{sub_dir}_{high_freq}kHz_stegosaurus.pdf'))
#
# '''Fit pn'''
# refTraces.pn_bar_plot(plot=True, normalised=True)
# plt.xlabel('Photon number')
# plt.ylabel('Counts')
# plt.savefig(DFUtils.create_filename(save_dir + rf'\{sub_dir}_100kHz_pn_plot.pdf'))
#
# highTraces.pn_bar_plot(plot=True, normalised=True)
# plt.xlabel('Photon number')
# plt.ylabel('Counts')
# plt.savefig(DFUtils.create_filename(save_dir + rf'\{sub_dir}_{high_freq}_pn_plot.pdf'))

'''Traces'''
to_show = 1000
data100_to_plot = data100[:2*to_show].reshape((to_show, 2*500))
data_high_to_plot = data_high[:2*to_show].reshape((to_show, 2*period))

fontsize = 14
# plot together
fig, axs = plt.subplot_mosaic([['(a) 100kHz', f'(b) {high_freq}kHz']],
                              figsize=(10,3), layout='constrained')
# plt.subplots_adjust(hspace=0.5, wspace=0.)

for label, ax in axs.items():
    ax.set_title(label, fontfamily='serif', loc='left', fontsize=fontsize+2)

ax1 = axs['(a) 100kHz']
for i in range(to_show):
    ax1.plot(np.arange(data100_to_plot.shape[1])*20/1000, data100_to_plot[i],alpha=0.05)
ax1.set_xlabel(r'$\mu s$', fontsize=fontsize)
ax1.set_ylabel(r'$\mu V$', fontsize=fontsize)
ax1.set_ylim(-1000, 20000)
ax1.set_xlim(0, 20)
ax1.set_xticks(np.arange(0, 20, 5))
ax1.tick_params(labelsize=fontsize-2)

ax2 = axs[f'(b) {high_freq}kHz']
for i in range(to_show):
    ax2.plot(np.arange(data_high_to_plot.shape[1])*20/1000, data_high_to_plot[i], alpha=0.05)
ax2.set_xlabel(r'$\mu s$', fontsize=fontsize)
ax2.set_ylim(-1000, 20000)
ax2.set_yticks([])
ax2.set_xlim(0, 2*period*20/1000)
ax2.tick_params(labelsize=fontsize-2)

# ax3 = axs['(c)']
# ax4 = axs['(d)']
#
# refTraces.fit_histogram(plot=True, fig_name=None, coloring=False, indexing=True,
#                         hist_color='dimgray', plot_fit=False)
# plt.ylim(0, 320)
# pos3 = ax3.get_position()
# ax3.set_position(ax4.get_position())
# ax4.set_position(pos3)
#
# # ax3.hist(refTraces.inner_products(), bins=refTraces.num_bins, color='cyan', alpha=1.)
# ax3.set_xlabel('Inner product', fontsize=fontsize)
# ax3.set_ylabel('Counts', fontsize=fontsize)
# ax3.set_ylim(0, 320)
# ax3.tick_params(labelsize=fontsize-2)
#
# ax4.hist(highTraces.inner_products(), bins=highTraces.num_bins, color='dimgray', alpha=1.)
# ax4.set_xlabel('Inner product', fontsize=fontsize)
# ax4.set_ylim(0, 320)
# ax4.set_yticks([])
# ax4.tick_params(labelsize=fontsize-2)

plt.show()

fig.savefig(DFUtils.create_filename(save_dir + rf'\{sub_dir}_100_vs_{high_freq}kHz_just_traces.pdf'))