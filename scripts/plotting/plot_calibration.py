import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
from scipy.optimize import curve_fit
from scipy.stats import bootstrap
from src.utils import poisson_norm

'''Compares the power meter calibrated photon number distribution with what the TES reports at 100kHz '''

powers = np.arange(12)
modeltype='IP'
bar_width= 0.4

alphabet = list(string.ascii_lowercase)

params_dir = rf'..\..\Results\Tomography_data_2024_04\Params'
log_df = pd.read_csv(params_dir + r'\log_2024_04_22.csv')
result_time_stamp = r'2024-05-15(19-27-44.036789)'

fig1, axs1 = plt.subplot_mosaic(powers.reshape((4, 3)), sharex=True, sharey=True, figsize=(15, 10), layout='constrained')  # plot 100kHz pn distribution

pm_mean_pns = np.zeros(len(powers))
tes_mean_pns = np.zeros((len(powers), 3))  # with negative and postive errors
for i_power, power in enumerate(powers):

    # set up plot
    ax1 = axs1[power]
    ax1.set_title(f'({alphabet[i_power]}) power_{power}', loc='left')

    # TES result
    params_df = pd.read_csv(params_dir + rf'\{modeltype}\{modeltype}_results_power_{power}.csv')
    tes_distrib = list(params_df.loc[params_df['rep_rate']==100, '0':].iloc[0])
    labels = np.arange(len(tes_distrib))

    # bootstrap for mean photon number
    raw_labels = np.load(
        params_dir + rf'\..\{modeltype}\power_{power}_{result_time_stamp}\{modeltype}_power_{power}_100kHz_raw_labels.npy')
    print(f'Start bootstrapping for power_{power}')
    t1 = time.time()
    tes_mean = np.mean(raw_labels)
    btstrp_res = bootstrap((raw_labels,), np.mean, confidence_level=0.95, batch=100, method='basic')
    t2 = time.time()
    print(f'Finish after {t2 - t1}s')
    tes_mean_pns[i_power] = [tes_mean, tes_mean - btstrp_res.confidence_interval.low,
                             btstrp_res.confidence_interval.high - tes_mean]

    # power meter calibration result
    pm_mu =  log_df.loc[(log_df['power_group'] == f'power_{power}') & (log_df['rep_rate/kHz'] == 100), 'pm_estimated_av_pn'].iloc[0]
    pm_mean_pns[i_power] = pm_mu
    pm_distrib = poisson_norm(labels, pm_mu)

    ax1.bar(labels - bar_width/2, pm_distrib, width=bar_width, align='center', label='Input')
    ax1.bar(labels + bar_width / 2, tes_distrib, width=bar_width, align='center', label='Measured')

    if i_power == 0 :
        ax1.legend()

    ax1.set_xlabel('Photon number')
    ax1.set_ylabel('Probability')
    ax1.set_xticks(labels[::3])

fig2, ax2 = plt.subplots()  # plot input vs measured mean photon number

ax2.set_title('Detector efficiency')
pm_errors = pm_mean_pns * 0.058
ax2.errorbar(pm_mean_pns, tes_mean_pns[:, 0], xerr=pm_errors, yerr=tes_mean_pns[:, 1:].T, fmt= 'o', ls='None')
ax2.set_xlabel('Input average photon number (power meter)')
ax2.set_ylabel('Measured average photon number (TES 100kHz)')

def loss(x, eta):
    return eta * x

popt, pcov = curve_fit(loss, pm_mean_pns, tes_mean_pns[:, 0], p0=[0.9])

ax2.plot(pm_mean_pns, popt[0] * pm_mean_pns, ls='dashed')

plt.show()