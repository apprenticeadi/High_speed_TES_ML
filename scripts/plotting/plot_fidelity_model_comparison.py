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

models = ['IP', 'BDT', 'RF']

max_input = 16  # number of columns
max_detected = 16  # number of rows
trunc = max_detected
fontsize = 14

'''Load reference theta'''
ip_dir = DFUtils.return_filename_from_head(r'..\..\Results\Tomography_data_2024_04\tomography\witherror',
                                           rf'tomography_on_IP_{max_input}x{max_detected}')

ref_ip_rep_rate = 100  # reference inner product rep rate, to plot and to calculate fidelity against
ref_thetas = np.load(ip_dir + rf'\{ref_ip_rep_rate}kHz_estimated_thetas.npy')[:trunc+1, :trunc+1]
ref_theta = np.mean(ref_thetas, axis=0)

'''Plot Fidelities'''
rep_vals = np.arange(100, 900, 100)

fig2, ax2 = plt.subplots(figsize=(10, 4), layout='constrained')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i_model, ml_model in enumerate(models):
    rf_dir = DFUtils.return_filename_from_head(r'..\..\Results\Tomography_data_2024_04\tomography\witherror',
                                               rf'tomography_on_{ml_model}_{max_input}x{max_detected}')

    rf_fids = np.zeros((len(rep_vals), len(ref_thetas), trunc+1))
    for i_rep, rep_rate in enumerate(rep_vals):

        # rf thetas
        rf_thetas = np.load(rf_dir + rf'\{rep_rate}kHz_estimated_thetas.npy')[:trunc+1, :trunc+1]
        for i_repeat, theta_rec in enumerate(rf_thetas):
            rf_fids[i_rep, i_repeat, :] = fidelity_by_n(theta_rec, ref_theta)

    fids = np.mean(rf_fids, axis=2)  # average over n
    fid_means = np.mean(fids, axis=1)  # average over repeats
    fid_stds = np.std(fids, axis=1)

    yerrs = np.zeros((2, len(fid_stds)))
    for i_rep in range(len(fid_stds)):
        yerrs[0, i_rep] = np.min([fid_stds[i_rep], fid_means[i_rep]-0])
        yerrs[1, i_rep] = np.min([fid_stds[i_rep], 1 - fid_means[i_rep]])

    if ml_model == 'IP':
        result = {'ls': '--', 'alpha': 0.5}
    else:
        result = {'ls': '-', 'alpha': 0.8}

    ax2.errorbar(rep_vals, fid_means, yerr=yerrs, fmt='.', ls=result['ls'], alpha=result['alpha'], label=ml_model)

ax2.set_ylim(0,1)
ax2.set_ylabel('Fidelity', fontsize=fontsize)
ax2.set_xlabel('Rep rate/kHz', fontsize=fontsize)
ax2.tick_params(labelsize=fontsize-2)
ax2.legend(fontsize=fontsize, loc='lower left')
ax2.set_title('Tomography fidelity')
plt.show()


# plot_dir = rf'..\..\Plots\Tomography_data_2024_04\tomography\{max_input}x{max_detected}'
# fig.savefig(DFUtils.create_filename(plot_dir + rf'\{max_detected}x{max_input}_POVM_reconstruction.pdf'))
# fig2.savefig(plot_dir+rf'\{trunc}x{trunc}_av_fidelities.pdf')
