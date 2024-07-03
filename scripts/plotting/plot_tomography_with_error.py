import time
import numpy as np
import datetime
import pandas as pd
import logging
from scipy.special import factorial
import string

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize

from src.utils import DFUtils, LogUtils
from scripts.process_data.tomography import fidelity_by_n

ml_model = 'KNN'

max_input = 16  # number of columns
max_detected = 16  # number of rows

ip_dir = DFUtils.return_filename_from_head(r'..\..\Results\Tomography_data_2024_04\tomography\witherror',
                                           rf'tomography_on_IP_{max_input}x{max_detected}')
rf_dir = DFUtils.return_filename_from_head(r'..\..\Results\Tomography_data_2024_04\tomography\witherror',
                                           rf'tomography_on_{ml_model}_{max_input}x{max_detected}')

plot_dir = rf'..\..\Plots\Tomography_data_2024_04\tomography\{max_input}x{max_detected}'

ref_ip_rep_rate = 100  # reference inner product rep rate, to plot and to calculate fidelity against
rep_rates_to_plot = [ref_ip_rep_rate, 500, 800]

# thetas to plot
thetas_to_plot = []
thetas_to_plot.append(np.load(ip_dir + rf'\{ref_ip_rep_rate}kHz_estimated_thetas.npy'))
for rf_rep_rate in rep_rates_to_plot[1:]:
    thetas_to_plot.append(np.load(rf_dir + rf'\{rf_rep_rate}kHz_estimated_thetas.npy'))

fontsize = 14

'''Plot POVM'''
# fig = plt.figure(figsize=(20, 8))
fig, axs = plt.subplots(2, 3, height_ratios=[3, 1], sharex='row', sharey='row', layout='constrained', figsize=(10, 4))

gs = gridspec.GridSpec(4, 18)
width = 0.4  # 3d bar plot width
depth = 0.2  # 3d bar plot depth
alphabet = list(string.ascii_lowercase)

_x = np.arange(max_input + 1)
_y = np.arange(max_detected + 1)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

# Reference POVM at 100kHz from inner product
for i_theta, thetas in enumerate(thetas_to_plot):
    theta_mean = np.mean(thetas, axis=0)[:max_detected + 1, :max_input + 1]
    theta_std = np.std(thetas, axis=0)[:max_detected + 1, :max_input + 1]

    # ax1 = fig.add_subplot(gs[:3, i*6:i*6+6], projection='3d')
    # ax2 = fig.add_subplot(gs[3:, i*6:i*6+6])
    ax1 = axs[0, i_theta]
    ax2 = axs[1, i_theta]

    '''Plot theta'''
    # cmap = plt.get_cmap()
    # norm = Normalize(vmin=0., vmax=1.)
    # ax1.bar3d(x - width/2, y - depth/2, np.zeros_like(theta_mean).ravel(), width, depth, theta_mean.ravel(), shade=True,
    #           color=cmap(norm(theta_mean.ravel())), alpha=0.8)
    # ax1.set_zlim(0,1)
    pc = ax1.pcolormesh(_xx, _yy, theta_mean, vmin=0., vmax=1.)
    ax1.set_xticks(_x[::2])
    # ax1.set_xlabel('Input photons', fontsize=fontsize)
    # ax1.tick_params(axis='x', which='major', pad=-3, labelsize=fontsize-2)
    # ax1.xaxis.labelpad = -5
    ax1.set_yticks(_y[::2])
    # ax1.set_yticklabels(_y[::2], verticalalignment='baseline', horizontalalignment='left')
    if i_theta == 0:
        # ax1.set_ylabel('Detected photons', fontsize=fontsize)
        ax1.set_ylabel('n', fontsize=fontsize + 2)
    ax1.tick_params(labelsize=fontsize - 2)
    ax1.set_title(f'({alphabet[i_theta]}) {rep_rates_to_plot[i_theta]}kHz', loc='left', fontsize=fontsize + 2)

    '''Plot diagonal with errorbar'''
    diag_mean = np.diag(theta_mean)
    diag_std = np.diag(theta_std)
    yerrs = np.zeros((2, len(diag_std)))
    for i_photon in range(len(diag_std)):
        yerrs[0, i_photon] = np.min([diag_std[i_photon], diag_mean[i_photon] - 0.])
        yerrs[1, i_photon] = np.min([diag_std[i_photon], 1. - diag_mean[i_photon]])

    ax1.set_xticks(_y[::2])
    ax1.set_xlabel(r'$m$', fontsize=fontsize + 2)

    ax2.bar(_y, diag_mean, width=0.8, align='center', yerr=yerrs)
    ax2.set_xticks(_y[::2])
    ax2.set_xlabel(r'$m$', fontsize=fontsize + 2)

    ax2.set_ylim(0, 1.1)
    if i_theta == 0:
        ax2.set_ylabel(r'$\theta_{mm}$', fontsize=fontsize)

    ax2.tick_params(labelsize=fontsize - 2)

cbar = fig.colorbar(pc, ax=axs[0], pad=0.01)
cbar.set_label(r'$\theta_{nm}$', fontsize=fontsize + 2)

'''Plot Fidelities'''
rep_vals = np.arange(100, 900, 100)

fig2, ax2 = plt.subplots(figsize=(10, 2), layout='constrained')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

trunc = max_detected

ref_thetas = thetas_to_plot[0][:trunc+1, :trunc+1]
ref_theta = np.mean(ref_thetas, axis=0)

ip_fids = np.zeros((len(rep_vals), len(ref_thetas), trunc+1))
rf_fids = np.zeros_like(ip_fids)
for i_rep, rep_rate in enumerate(rep_vals):

    # inner product thetas
    ip_thetas = np.load(ip_dir + rf'\{rep_rate}kHz_estimated_thetas.npy')[:trunc+1, :trunc+1]
    for i_repeat, theta_rec in enumerate(ip_thetas):
        ip_fids[i_rep, i_repeat, :] = fidelity_by_n(theta_rec, ref_theta)

    # rf thetas
    rf_thetas = np.load(rf_dir + rf'\{rep_rate}kHz_estimated_thetas.npy')[:trunc+1, :trunc+1]
    for i_repeat, theta_rec in enumerate(rf_thetas):
        rf_fids[i_rep, i_repeat, :] = fidelity_by_n(theta_rec, ref_theta)

res_dict = {
    'ip': {'fid': ip_fids, 'ls': 'dashed', 'alpha': 0.5, 'label': 'Inner Product', 'color': 'gray'},
    'rf': {'fid': rf_fids, 'ls': 'solid', 'alpha': 0.8, 'label': ml_model, 'color': 'black'}
}

for result in res_dict.values():
    fids = np.mean(result['fid'], axis=2)  # average over n
    fid_means = np.mean(fids, axis=1)  # average over repeats
    fid_stds = np.std(fids, axis=1)

    yerrs = np.zeros((2, len(fid_stds)))
    for i_rep in range(len(fid_stds)):
        yerrs[0, i_rep] = np.min([fid_stds[i_rep], fid_means[i_rep]-0])
        yerrs[1, i_rep] = np.min([fid_stds[i_rep], 1 - fid_means[i_rep]])

    ax2.errorbar(rep_vals, fid_means, yerr=yerrs, fmt='.', ls=result['ls'], alpha=result['alpha'], label=result['label'],
                 color=result['color']
                 )

ax2.set_ylim(0,1)
ax2.set_ylabel('Fidelity', fontsize=fontsize)
ax2.set_xlabel('Rep rate/kHz', fontsize=fontsize)
ax2.tick_params(labelsize=fontsize-2)
ax2.legend(fontsize=fontsize, loc='lower left')

plt.show()

# fig.savefig(DFUtils.create_filename(plot_dir + rf'\{max_detected}x{max_input}_POVM_reconstruction.pdf'))
# fig2.savefig(plot_dir+rf'\{trunc}x{trunc}_av_fidelities.pdf')
