import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import factorial

from src.utils import DFUtils, poisson_norm, tvd
import tes_resolver.config as config

powers = np.arange(0, 12, 1)
rep_rates = np.arange(100, 1100, 100)
rep_rate_mosaic = rep_rates.reshape((2,5))
modeltype='IP'
cal_rep_rate = 100

params_dir = config.home_dir + rf'\..\Results\Tomography_data_2024_04\Params\{modeltype}'

plot_dir = config.home_dir + rf'\..\Results\Tomography_data_2024_04\{modeltype}_distributions'

for power in powers:

    df = pd.read_csv(params_dir + rf'\{modeltype}_results_power_{power}.csv')

    # read in calibration data (100kHz)
    cal_distrib = np.array(df.loc[df['rep_rate'] == int(cal_rep_rate), '0':].iloc[0])
    pn_labels = np.arange(len(cal_distrib))

    # prepare figure to plot distributions
    fig, axs = plt.subplot_mosaic(rep_rate_mosaic, figsize=(20, 8), sharex=True, sharey=True, layout='constrained')

    mean_mus = np.zeros(len(rep_rates))
    rel_tvds = np.zeros(len(rep_rates))  # tvd with 100kHz distribution
    fit_tvds = np.zeros(len(rep_rates))

    for i, rep_rate in enumerate(rep_rates):
        # the distribution predicted by the classifier
        tes_distrib = np.array(df.loc[df['rep_rate'] == int(rep_rate), '0':].iloc[0])
        tes_distrib = np.nan_to_num(tes_distrib)  # convert nan to numbers

        # mean photon number
        mean_mu = np.sum(pn_labels * tes_distrib)
        mean_mus[i] = mean_mu

        # relative tvd
        rel_tvds[i] = tvd(tes_distrib, cal_distrib)

        # tvd with fitted poissonian function
        fit_distrib = poisson_norm(pn_labels, mean_mu)
        fit_tvds[i] = tvd(tes_distrib, fit_distrib)

        # plot pn distribution
        ax = axs[rep_rate]
        ax.bar(pn_labels, tes_distrib, width=0.5, align='center')
        ax.set_title(f'{rep_rate}kHz', loc='left')
        ax.set_ylabel('Probability')
        ax.set_xlabel('Photon number')
        ax.plot(pn_labels, fit_distrib, 'ro-', label=f'$\mu=${mean_mu:.2g}')

        ax.legend()

    fig.savefig(DFUtils.create_filename(plot_dir + rf'\power_{power}\pn_distributions.pdf'))

    fig2, axs2 = plt.subplots(3,1, figsize=(12, 10), sharex=True, sharey=False, layout='constrained')

    ax = axs2[0]  # mean photon
    ax.plot(rep_rates, mean_mus, 'o-')
    ax.set_ylabel('Mean photon number')
    ax.set_title('Mean photon number')

    ax = axs2[1]  # relative tvd with 100kHz
    ax.plot(rep_rates, rel_tvds, 'o-')
    ax.set_ylabel('TVD')
    ax.set_title(f'TVD with IP {cal_rep_rate}kHz')

    ax = axs2[2]  # tvd with fitted Poissonian
    ax.plot(rep_rates, fit_tvds, 'o-')
    ax.set_ylabel('TVD')
    ax.set_title(f'TVD with mean photon Poissonian fit')
    ax.set_xticks(rep_rates)
    ax.set_xlabel('Repetition rate/kHz')

    fig2.savefig(DFUtils.create_filename(plot_dir + rf'\power_{power}\means_and_tvds.pdf'))

    mean_tvd_df = pd.DataFrame({
        'rep_rate': rep_rates,
        'means': mean_mus,
        f'tvd_with_{cal_rep_rate}kHz': rel_tvds,
        'tvd_with_mean_poisson': fit_tvds
    })

    mean_tvd_df.to_csv(plot_dir + rf'\power_{power}\means_and_tvds.csv', index=False)