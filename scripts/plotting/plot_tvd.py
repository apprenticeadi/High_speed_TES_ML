import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import factorial

from src.utils import DFUtils, poisson_norm, tvd


powers = [1, 5, 8, 10]
rep_rates = np.arange(100, 900, 100)
modeltype='RF'

special_power = 5
special_reprate = 700
special_reprate2 = 800

params_dir = rf'..\..\Results\Tomography_data_2024_04\Params'
log_df = pd.read_csv(params_dir + r'\log_2024_04_22.csv')


results_dict = {}
for power in powers:
    df = pd.read_csv(params_dir + rf'\{modeltype}\{modeltype}_results_power_{power}.csv')
    df = df.sort_values('rep_rate')
    results_dict[power] = df

fontsize=12
fig, axs = plt.subplot_mosaic([['(a)', '(b)']], figsize=(15, 8), layout='constrained')
# left, bottom, width, height = [0.2, 0.4, 0.4, 0.45]
# ax2 = fig.add_axes([left, bottom, width, height])

# plt.subplots_adjust(wspace=0.3, hspace=0.1)

for label, ax in axs.items():
    ax.set_title(label, fontfamily='serif', loc='left', fontsize=fontsize+2)

ax1 = axs['(a)']
ax2 = axs['(b)']

mean_photons = {}
for power in powers:

    df = results_dict[power]

    tvds = np.zeros(len(rep_rates))

    mean_photon = 0

    for i, rep_rate in enumerate(rep_rates):
        tes_distrib = np.array(df.loc[df['rep_rate'] == int(rep_rate), '0':].iloc[0])
        tes_distrib = np.nan_to_num(tes_distrib)  # nan to 0s.
        labels = np.arange(len(tes_distrib))

        # power meter mean photon number
        pm_mu = log_df.loc[(log_df['power_group'] == f'power_{power}') & (log_df['rep_rate/kHz'] == rep_rate), 'pm_estimated_av_pn'].iloc[0]
        pm_distrib = poisson_norm(labels, pm_mu)

        tvds[i] = tvd(tes_distrib, pm_distrib)
        if i == 0:
            mean_photon = pm_mu
            mean_photons[power] = mean_photon

    ax1.plot(rep_rates, tvds, 'o-', label=f'{mean_photon:.2f}', alpha=0.8, markersize=8)

    if power == special_power:
        id_100 = np.argmax(rep_rates==100)
        id_special = np.argmax(rep_rates==special_reprate)
        id_special2 = np.argmax(rep_rates==special_reprate2)
        ax1.plot(rep_rates[id_100], tvds[id_100], 'D', color='black', zorder=4, markersize=12)
        ax1.plot(rep_rates[id_special], tvds[id_special], 'D', color='gray', zorder=4, markersize=12)
        ax1.plot(rep_rates[id_special2], tvds[id_special2], 'D', color='saddlebrown', zorder=4, markersize=12)

ax1.legend(loc='upper left', fontsize=fontsize-2)
ax1.set_xlabel('Repetition rate/kHz', fontsize=fontsize)
ax1.set_ylabel('TVD', fontsize=fontsize)
ax1.tick_params(labelsize=fontsize-2)


'''Plot special '''
# fig2, ax2 = plt.subplots(figsize=(12, 6))

df = results_dict[special_power]
row_100 = df.loc[df['rep_rate']==100, :]

pn_100= np.array(row_100.loc[:, '0':].iloc[0])
pn = np.array(df.loc[df['rep_rate']== special_reprate, '0':].iloc[0])
pn2 = np.array(df.loc[df['rep_rate']== special_reprate2, '0':].iloc[0])
labels = np.arange(len(pn_100))
width=0.3

ax2.bar(labels - width, pn_100, width=width, align='center', alpha=0.8, label='100kHz', color='black')
ax2.bar(labels, pn, width=width, align='center', alpha=0.8, label=f'{special_reprate}kHz', color='gray')
ax2.bar(labels + width, pn2, width=width, align='center', alpha=0.8, label=f'{special_reprate2}kHz', color='saddlebrown')

ax2.set_ylim(0, 0.5)
ax2.set_ylabel('Probability', fontsize=fontsize)
ax2.set_xlabel('Photon number', fontsize=fontsize)
ax2.tick_params(labelsize=fontsize-2)

# def poisson_norm(x, mu):
#     return (mu ** x) * np.exp(-mu) / factorial(x)

ax2.legend(loc='upper right', fontsize=fontsize-2)
plt.show()

# fig.savefig(DFUtils.create_filename(r'..\..\Plots\TVD_plots\TVD_plot_sidebyside_3specials.pdf'))


# plt.figure('accuracy')
# for power in powers:
#     df = results_dict[power]
#
#     freqs = np.array(df.loc[:, 'rep_rate'])
#     accuracies = np.array(df.loc[:, 'acc_score'])
#     pn_100 = np.array(df.loc[df['rep_rate'] == 100, '0':].iloc[0])
#     labels = np.arange(len(pn_100))
#     mean_photon = np.sum(pn_100 * labels)
#
#     plt.plot(freqs, accuracies, 'o-', label=f'{mean_photon:.2f}', alpha=0.8, markersize=8)
#
# plt.legend(fontsize=fontsize-2)
# plt.ylabel('Test accuracy score', fontsize=fontsize)
# plt.xlabel('Repetition rate/kHz', fontsize=fontsize)
# plt.tick_params(labelsize = fontsize-2)
# plt.savefig(r'..\..\Plots\TVD_plots\RF_accuracy_scores.pdf')
