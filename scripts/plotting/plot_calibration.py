import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
from scipy.optimize import curve_fit

from src.utils import poisson_norm

'''Compares the power meter calibrated photon number distribution with what the TES reports at 100kHz '''

powers = np.arange(12)
modeltype='RF'
bar_width= 0.4

alphabet = list(string.ascii_lowercase)

params_dir = rf'..\..\Results\Tomography_data_2024_04\Params'
log_df = pd.read_csv(params_dir + r'\log_2024_04_22.csv')

fig1, axs1 = plt.subplot_mosaic(powers.reshape((4, 3)), sharex=True, sharey=True, figsize=(15, 10), layout='constrained')  # plot 100kHz pn distribution
fig2, ax2 = plt.subplots()  # plot input vs measured mean photon number

pm_mean_pns = np.zeros(len(powers))
tes_mean_pns = np.zeros(len(powers))
for i_power, power in enumerate(powers):

    # set up plot
    ax1 = axs1[power]
    ax1.set_title(f'({alphabet[i_power]}) power_{power}', loc='left')

    # TES result
    params_df = pd.read_csv(params_dir + rf'\{modeltype}\{modeltype}_results_power_{power}.csv')
    tes_distrib = list(params_df.loc[params_df['rep_rate']==100, '0':].iloc[0])
    labels = np.arange(len(tes_distrib))
    tes_mean_pns[i_power] = np.sum(np.array(tes_distrib) * labels)

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

ax2.set_title('Detector efficiency')
ax2.plot(pm_mean_pns, tes_mean_pns, 'o', ls='None')
ax2.set_xlabel('Measured average photon number')
ax2.set_ylabel('Input average photon number')

def loss(x, eta):
    return eta * x

popt, pcov = curve_fit(loss, pm_mean_pns, tes_mean_pns, p0=[0.9])

ax2.plot(pm_mean_pns, popt[0] * pm_mean_pns, ls='dashed')

plt.show()