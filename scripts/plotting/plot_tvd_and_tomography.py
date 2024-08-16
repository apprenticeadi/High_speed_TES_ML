import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import string

plt.show()
plt.ion()

alphabet = list(string.ascii_lowercase)
results_dir = r'..\..\Plots\Tomography_data_2024_04\tvd_and_tomography'
fontsize = 14
title_params = {'fontfamily': 'serif', 'loc': 'left', 'fontsize': fontsize + 2}

fig, axs = plt.subplot_mosaic(mosaic=
                              '''
                              aaabbbccc
                              aaabbbccc
                              dddeeefff
                              dddeeefff
                              dddeeefff
                              ggghhhiii
                              .jjjjjjj.
                              .jjjjjjj.
                              ''',
                              figsize=(21, 10), layout='constrained')

'''plot tvd'''
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
swapped_color_cycle = [color_cycle[0], color_cycle[2], color_cycle[1], color_cycle[3]]
models = ['IP', 'CNN', 'KNN', 'HDBSCAN']
color_dict = {}
for i_model, model in enumerate(models):
    color_dict[model] = swapped_color_cycle[i_model]

for i_power, power in enumerate([0, 6, 9]):
    tvd_df = pd.read_csv(results_dir + rf'\tvd_power_{power}.csv')
    mu = tvd_df.loc[0, 'mu']
    rep_rates = np.array(tvd_df['rep_rate'])

    index = alphabet[i_power]
    ax = axs[index]
    ax.set_title(fr'({index}) ' + r'$\mu=$' + f'{mu:.2f}', **title_params)

    for i_model, model in enumerate(models):
        tvds = tvd_df[model]
        args_not_nan = np.argwhere(~np.isnan(tvds)).flatten()
        if model == 'IP':
            style = 'o--'
            alpha=0.5
        elif model == 'HDBSCAN':
            style = 'o:'
            alpha=0.5
        else:
            style='o-'
            alpha=0.8
        ax.plot(rep_rates[args_not_nan], tvds[args_not_nan], style, alpha=alpha,
                label=model, color=color_dict[model])

    ax.set_xlim(50, 1050)
    ax.set_xticks(np.arange(100, 1100, 100))
    ax.set_xlabel('Repetition rate (kHz)', fontsize=fontsize)
    ax.set_yscale('linear')
    ax.set_ylim(-0.1, 1.)
    if i_power == 0:
        ax.set_ylabel('TVD', fontsize=fontsize)
        ax.legend(fontsize=fontsize-2, loc='upper left')
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=fontsize-2)

'''Plot POVM'''
models = ['IP', 'KNN', 'KNN']
rep_rates = [100, 500, 800]
for i_rep, rep_rate in enumerate(rep_rates):
    mean_df = pd.read_csv(results_dir + rf'\{models[i_rep]}_{rep_rate}kHz_mean_theta.csv')
    mean_theta = mean_df.to_numpy()[:, 1:]
    _x = mean_df.columns[1:].to_numpy().astype(int)
    _y = mean_df.index.to_numpy().astype(int)
    _xx, _yy = np.meshgrid(_x, _y)

    std_df = pd.read_csv(results_dir + rf'\{models[i_rep]}_{rep_rate}kHz_std_theta.csv')
    std_theta = std_df.to_numpy()[:, 1:]

    '''Plot theta'''
    index = alphabet[i_rep + 3]
    ax = axs[index]
    ax.set_title(fr'({index}) {models[i_rep]} {rep_rate}kHz', **title_params)

    pc = ax.pcolormesh(_xx, _yy, mean_theta, vmin=0., vmax=1.)
    ax.set_xticks(_x[::2])
    ax.set_yticks(_y[::2])
    if i_rep == 0:
        ax.set_ylabel('n', fontsize=fontsize)
    else:
        ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=fontsize-2)

    '''Plot diagonal with errorbar'''
    diag_mean = np.diag(mean_theta)
    diag_std = np.diag(std_theta)
    yerrs = np.zeros((2, len(diag_std)))
    for i_photon in range(len(diag_std)):
        yerrs[0, i_photon] = np.min([diag_std[i_photon], diag_mean[i_photon] - 0.])
        yerrs[1, i_photon] = np.min([diag_std[i_photon], 1. - diag_mean[i_photon]])

    ax = axs[alphabet[i_rep + 6]]
    ax.bar(_y, diag_mean, width=0.8, align='center', yerr=yerrs, color=color_dict[models[i_rep]])
    ax.set_xticks(_y[::2])
    ax.set_xlabel(r'$m$', fontsize=fontsize)
    ax.set_ylim(0, 1.1)
    if i_rep == 0:
        ax.set_ylabel(r'$\theta_{mm}$', fontsize=fontsize)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=fontsize-2)

cbar = fig.colorbar(pc, ax=axs['f'], pad=0.01)
cbar.set_label(r'$|\theta_{nm}|$', fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize-2)

'''Plot fidelity'''
ax = axs['j']
ax.set_title('(f) POVM Fidelity', **title_params)

models = ['IP', 'KNN']
for model in models:
    f_df = pd.read_csv(results_dir + rf'\{model}_fidelities.csv')
    if model == 'IP':
        ls = 'dashed'
        alpha=0.5
    else:
        ls = 'solid'
        alpha=0.8
    ax.errorbar(f_df['rep_rate'], f_df['fidelity'],
                yerr=np.vstack((f_df['n_error'], f_df['p_error'])),
                label=model, ls=ls, alpha=alpha, color=color_dict[model])

    ax.set_ylim(0, 1)
    ax.set_ylabel('Fidelity', fontsize=fontsize)
    ax.set_xlabel('Repetition rate (kHz)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-2)
    ax.legend(fontsize=fontsize-2, loc='lower left')


plt.show()
plt.pause(10)

fig.savefig(results_dir + r'\tvd_and_tomography_example_plot.pdf')

