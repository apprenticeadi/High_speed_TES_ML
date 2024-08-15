import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import string

matplotlib.use('TkAgg')

'''Plot the pca data from tabular classification'''
data_group = 'power_6'
save_dir = rf'../../Plots/Tomography_data_2024_04/trace_pca_plots/{data_group}'

rep_rates = [100, 500, 800]

'''Parameters'''
fontsize = 14
alphabet = list(string.ascii_lowercase)
title_params = {'fontfamily': 'serif', 'loc': 'left', 'fontsize': fontsize + 2}

fig, axs = plt.subplots(4, len(rep_rates), sharey='row', layout='constrained', figsize=(15, 12))

for i_rep, rep_rate in enumerate(rep_rates):
    rep_dir = save_dir + rf'\{rep_rate}kHz'

    '''Plot raw traces'''
    ax = axs[0, i_rep]
    ax.set_title(f'({alphabet[i_rep]}) {rep_rate}kHz', **title_params)

    raw_df = pd.read_csv(rep_dir + r'\first_1000traces.csv')

    x = raw_df['time/us']
    for i in range(1000):
        y = raw_df[f'{i}']
        ax.plot(x, y, alpha=0.05)

    if i_rep == 0:
        ax.set_ylabel(r'$mV$', fontsize=fontsize)
    ax.set_xlabel(r'$\mu s$', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-2)

    '''Plot ip stegosaurus'''
    ax = axs[1, i_rep]
    ax.set_title(f'({alphabet[i_rep + 1 * len(rep_rates)]})', **title_params)

    steg_df = pd.read_csv(rep_dir + r'\ip_stegosaurus.csv')
    heights = list(steg_df['num_traces'])[:-1]
    bin_edges = list(steg_df['ip_bin_edges'])

    ax.bar(bin_edges[:-1], heights, width=np.diff(bin_edges), align='edge', alpha=0.8, color='gray')

    if i_rep == 0:
        ax.set_ylabel(r'Counts', fontsize=fontsize)
    ax.set_xlabel(r'Inner product', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)

    '''Plot principal components'''
    ax = axs[2, i_rep]
    ax.set_title(f'({alphabet[i_rep + 2 * len(rep_rates)]})', **title_params)

    pc_df = pd.read_csv(rep_dir + r'\principal_components.csv')
    x = pc_df['time']
    for i in range(1, 3):
        y = pc_df[f'PC{i}']
        ax.plot(x, y, label=f'PC{i}')

    if i_rep == 0:
        ax.set_ylabel('arb. unit', fontsize=fontsize)
    ax.set_xlabel(r'$\mu s$', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-2)
    ax.legend(loc='upper right', fontsize=fontsize-2)

    '''Plot pca factor scores density scatter'''
    ax = axs[3, i_rep]
    ax.set_title(f'({alphabet[i_rep + 3 * len(rep_rates)]})', **title_params)

    f_df = pd.read_csv(rep_dir + r'\density_scatter_of_first_2_factor_scores.csv')
    f1 = f_df['F1']
    f2 = f_df['F2']
    z = f_df['interpn_num_points']

    image = ax.scatter(f1, f2, c=z, s=5, cmap='viridis')
    cbar = fig.colorbar(image, ax=ax, location='right', pad=0.01)
    cbar.ax.set_ylabel('Number of points', fontsize=fontsize-2)
    ax.set_xlabel(r'$F_1$', fontsize=fontsize)
    ax.set_ylabel(r'$F_2$', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)


plt.show()
plt.pause(10)

fig.savefig(save_dir + rf'\trace_pca_example_plot.pdf')

