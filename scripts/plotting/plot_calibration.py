import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
from utils.utils import poisson_norm

import scipy.odr as odr

'''Compares the power meter calibrated photon number distribution with what the TES reports at 100kHz '''
powers = np.arange(12)
modeltype='IP'
bar_width= 0.8
rep_rate = 100

pm_mean_error = 0.046
font_size = 14

alphabet = list(string.ascii_lowercase)

params_dir = rf'..\..\Results\Tomography_data_2024_04\Params'
log_df = pd.read_csv(params_dir + r'\log_2024_04_22.csv')

'''Plot every single power distribution compared with PM Poissonian distribution'''
fig1, axs1 = plt.subplot_mosaic(powers.reshape((4, 3)), sharex=True, sharey=True, figsize=(10,8), layout='constrained')  # plot 100kHz pn distribution

pm_mean_pns = np.zeros(len(powers))
tes_mean_pns = np.zeros((len(powers), 3))  # with negative and postive errors
for i_power, power in enumerate(powers):

    # Set up plot
    ax1 = axs1[power]
    ax1.set_title(f'({alphabet[i_power]}) power_{power}', loc='left', fontsize=font_size-2)

    # TES result with errors
    distrib_df = pd.read_csv(params_dir + rf'\{modeltype}\{modeltype}_results_power_{power}.csv')
    tes_distrib = np.array(distrib_df.loc[distrib_df['rep_rate'] == rep_rate, '0':].iloc[0])
    tes_distrib = np.nan_to_num(tes_distrib)

    labels = np.arange(len(tes_distrib))

    tes_distrib_errors = np.zeros((2, len(tes_distrib)))  # first row negative, second row positive
    for i_row, n_or_p in enumerate(['n', 'p']):
        errors_df = pd.read_csv(params_dir + rf'\{modeltype}\bootstrapped\{modeltype}_results_power_{power}_{n_or_p}_error.csv')
        tes_distrib_errors[i_row] = np.array(errors_df.loc[errors_df['rep_rate'] == rep_rate, '0':].iloc[0])
    tes_distrib_errors = np.nan_to_num(tes_distrib_errors)

    # TES reported mean photon number and error
    mean_df = pd.read_csv(params_dir+rf'\{modeltype}\bootstrapped_means\{modeltype}_results_power_{power}.csv')
    tes_mean_pns[i_power] = np.array(mean_df.loc[mean_df['rep_rate'] == rep_rate, ['mean_pn', 'n_error', 'p_error']].iloc[0])

    # power meter calibration result with errors
    pm_mu =  log_df.loc[(log_df['power_group'] == f'power_{power}') & (log_df['rep_rate/kHz'] == 100), 'pm_estimated_av_pn'].iloc[0]
    pm_mean_pns[i_power] = pm_mu
    pm_distrib = poisson_norm(labels, pm_mu)

    pm_distrib_errors = np.abs(labels - pm_mu) * pm_mean_error * pm_distrib

    ax1.bar(labels, tes_distrib, width=bar_width, align='center', label='Measured',
            yerr = tes_distrib_errors, capsize=5, error_kw={'barsabove':True, 'elinewidth':10},
            color='blue', alpha=0.7)
    ax1.errorbar(labels, pm_distrib, yerr=pm_distrib_errors, fmt='.--', label='Input', color='red')

    if i_power == 0:
        ax1.legend(fontsize=font_size-2)
    if i_power % 3 == 0:
        ax1.set_ylabel('Probability', fontsize=font_size-2)
    if i_power // 3 == 3:
        ax1.set_xlabel('Photon number', fontsize=font_size-2)
    ax1.set_xticks(labels[::3])
    ax1.tick_params(labelsize=font_size-4)


fig2, ax2 = plt.subplots(figsize=(10, 3), layout='constrained')  # plot input vs measured mean photon number

# ax2.set_title('Detector efficiency')
pm_errors = pm_mean_pns * pm_mean_error
ax2.errorbar(pm_mean_pns, tes_mean_pns[:, 0], xerr=pm_errors, yerr=tes_mean_pns[:, 1:].T, fmt= '.', ls='None', label='Data')
ax2.set_xlabel('Input average photon number', fontsize=font_size)
ax2.set_ylabel(f'Measured average photon number', fontsize=font_size)
ax2.tick_params(labelsize=font_size-2)

def fit(B, x):
    return B[0] * x

linear = odr.Model(fit)
mydata = odr.Data(pm_mean_pns, tes_mean_pns[:, 0], wd=pm_errors, we=np.mean(tes_mean_pns[:, 1:], axis=1))
myodr = odr.ODR(mydata, linear, beta0=[0.93])
myoutput = myodr.run()

ax2.plot(pm_mean_pns, myoutput.beta[0] * pm_mean_pns, ls='dashed', label=f'{myoutput.beta[0]*100:.3g} efficiency')
ax2.legend(fontsize=font_size-2, loc='upper left')

plt.show()

myoutput.pprint()