import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import string

matplotlib.use('TkAgg')

'''Plot the pca data from tabular classification'''
data_group = 'power_6'
save_dir = rf'../../Plots/Tomography_data_2024_04/ml_and_pca/{data_group}'

'''Parameters'''
fontsize = 14
alphabet = list(string.ascii_lowercase)
title_params = {'fontfamily': 'serif', 'loc': 'left', 'fontsize': fontsize + 2}
color_cycle = list(plt.get_cmap('tab20').colors)
color_cycle = color_cycle[0::2] + color_cycle[1::2]

rep_rates = [500, 800]

# cal distribution
cal_df = pd.read_csv(save_dir + r'\calibration_pn_distribution_by_IP_100kHz.csv')

fig, axs = plt.subplots(4, 3, layout='constrained', figsize=(21, 10), height_ratios=[3,2,3,2],
                        sharex='row')
axs[1, 0].axis('off')
axs[3, 0].axis('off')

for i_rep, rep_rate in enumerate(rep_rates):
    rep_dir = save_dir + rf'\{rep_rate}kHz'

    '''Scatter plot pca'''
    idx = alphabet[i_rep*5+0]
    ax = axs[i_rep*2, 0]  # axs[idx]
    ax.set_title(f'({idx}) {rep_rate}kHz', **title_params)

    f_df = pd.read_csv(rep_dir + r'\density_scatter_of_first_2_factor_scores.csv')
    f1 = f_df['F1']
    f2 = f_df['F2']
    z = f_df['interpn_num_points']

    image = ax.scatter(f1, f2, c=z, s=5, cmap='viridis')
    # cbar = fig.colorbar(image, ax=ax, location='right', pad=0.01)
    # cbar.ax.set_ylabel('Number of points', fontsize=fontsize - 2)
    # cbar.ax.tick_params(labelsize=fontsize - 2)

    '''Scatter plot KNN pca results'''
    idx = alphabet[i_rep*5+1]
    ax = axs[i_rep*2, 1]  # axs[idx]
    ax.set_title(f'({idx}) KNN', **title_params)

    f_df = pd.read_csv(rep_dir + r'\KNN_factor_scores.csv')
    f1s = f_df['F1']
    f2s = f_df['F2']
    knn_labels = f_df['KNN_label']

    for pn in set(knn_labels):
        indices = np.argwhere(knn_labels == pn).ravel()
        indices = indices[:200]
        ax.scatter(f1s[indices], f2s[indices], alpha=0.8, s=5,
                   color=color_cycle[pn % len(color_cycle)])

    '''Plot KNN pn distribution'''
    idx = alphabet[i_rep*5+3]
    ax = axs[i_rep*2+1, 1]  # axs[idx]
    ax.set_title(f'({idx})', **title_params)

    d_df = pd.read_csv(rep_dir + r'\KNN_pn_distribution.csv')
    pns = d_df['pn']
    pn_distrib = d_df['probability']

    for i_pn, pn in enumerate(pns):
        pn = int(pn)
        ax.bar(pn, pn_distrib[i_pn], width=0.8, align='center', alpha=0.8, color=color_cycle[pn % len(color_cycle)])

    ax.errorbar(cal_df['pn'], cal_df['probability'],
                yerr=np.vstack((cal_df['n_error'], cal_df['p_error'])),
                fmt='.', ls='--',
                color='red', label='Calibration')

    '''Scatter plot HDBSCAN cluster results'''
    idx = alphabet[i_rep * 5 + 2]
    ax = axs[i_rep*2, 2]  # axs[idx]
    ax.set_title(f'({idx}) HDBSCAN', **title_params)

    cluster_df = pd.read_csv(rep_dir + r'\HDBSCAN_factor_scores.csv')
    f1s = cluster_df['F1']
    f2s = cluster_df['F2']
    cluster_labels = cluster_df['cluster_label']

    for cluster_label in set(cluster_labels):
        indices = np.argwhere(cluster_labels == cluster_label).ravel()
        if cluster_label == -1:
            ax.scatter(f1s[indices], f2s[indices], alpha=0.5, s=0.5,
                       color='black', label='Unclassified')
        else:
            indices = indices[:200]
            ax.scatter(f1s[indices], f2s[indices], alpha=0.8, s=5,
                       color=color_cycle[ (cluster_label) % len(color_cycle)])

    '''Plot HDBSCAN distribution'''
    idx = alphabet[i_rep*5+4]
    ax = axs[i_rep*2+1, 2]  # axs[idx]
    ax.set_title(f'({idx})', **title_params)

    cluster_distrib_df = pd.read_csv(rep_dir + r'\HDBSCAN_pn_distribution.csv')
    pns = cluster_distrib_df['pn']
    cluster_label_list = cluster_distrib_df['cluster_label']
    pn_distrib = cluster_distrib_df['probability']

    bottoms = np.zeros(np.max(pns) + 2)  # first one is for -1
    for i_pn, pn in enumerate(pns):
        if pn == -1:
            color = 'black'
        else:
            cluster_label = cluster_label_list[i_pn]
            color = color_cycle[ (cluster_label) % len(color_cycle)]
        ax.bar(pn, pn_distrib[i_pn], width=0.8, align='center', alpha=0.8, color=color, bottom=bottoms[pn+1])
        bottoms[pn+1] += pn_distrib[i_pn]

    ax.errorbar(cal_df['pn'], cal_df['probability'],
                yerr=np.vstack((cal_df['n_error'], cal_df['p_error'])),
                fmt='.', ls='--',
                color='red', label='Calibration')

# formatting
for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        ax = axs[i, j]

        if i == 0 or i == 2:
            ax.set_xlim(-70000, 120000)
            ax.set_ylim(-35000, 35000)
            ax.set_xticks([-50000, 0, 50000, 100000])
            ax.set_yticks([-20000, 0, 20000])
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_fontsize(fontsize - 2)

            ax.set_xlabel(r'$F_1$', fontsize=fontsize)
            if j == 0:
                ax.set_ylabel(r'$F_2$', fontsize=fontsize)
                # ax.set_yticklabels([-20000, 0, 20000])
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                ax.yaxis.get_offset_text().set_fontsize(fontsize-2)
            else:
                ax.set_yticklabels([])
            if j == 2:
                ax.legend(loc='upper right', fontsize=fontsize - 2)

        if i == 1 or i == 3:
            ax.set_ylim(0, 0.25)
            ax.set_xlim(-2, 15)
            ax.set_yticks([0., 0.1, 0.2])
            ax.set_xticks(np.arange(-1, 15))
            ax.set_xlabel('Photon number', fontsize=fontsize)

            if j == 1:
                ax.set_yticklabels([0., 0.1, 0.2])
                ax.set_ylabel('Probability', fontsize=fontsize)
                ax.legend(loc='upper right', fontsize=fontsize - 2)
            else:
                ax.set_yticklabels([])

        ax.tick_params(labelsize=fontsize - 2)


plt.show()
plt.pause(10)

fig.savefig(save_dir + rf'\ml_and_pca_example_plot.pdf')


