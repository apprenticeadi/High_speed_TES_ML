import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import factorial

from src.utils import DFUtils, poisson_norm, tvd

powers = [1, 6, 10]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

special_power = 6
special_rep_rates = [500, 800]
special_colors = ['gray', 'saddlebrown']

rep_rates = np.arange(100, 1100, 100)

params_dir = rf'..\..\Results\Tomography_data_2024_04\Params'
# log_df = pd.read_csv(params_dir + r'\log_2024_04_22.csv')
# compare_with_pm = False


fontsize = 12
fig, axs = plt.subplot_mosaic([['(a)', '(b)']], figsize=(12, 4), layout='constrained')
for label, ax in axs.items():
    ax.set_title(label, fontfamily='serif', loc='left', fontsize=fontsize + 2)

ax1 = axs['(a)']
ax2 = axs['(b)']

# fig, ax1 = plt.subplots(figsize=(10, 6), layout='constrained')
special_distribs = {}
ref_mean_photons = {}
for i_power, power in enumerate(powers):
    color = colors[i_power]

    # IP and RF data
    df_ip = pd.read_csv(params_dir + rf'\IP\IP_results_power_{power}.csv')
    df_rf = pd.read_csv(params_dir + rf'\RF\RF_results_power_{power}.csv')
    tvds = np.zeros((2, len(rep_rates)))  # first row ip, second row rf

    # Calculate TVD for each rep rate
    ref_mu = 1.
    for i_rep, rep_rate in enumerate(rep_rates):
        # get IP distribution
        ip_distrib = np.array(df_ip.loc[df_ip['rep_rate'] == int(rep_rate), '0':].iloc[0])
        ip_distrib = np.nan_to_num(ip_distrib)  # nan to 0s.
        ip_labels = np.arange(len(ip_distrib))

        # update reference mean photon number from 100kHz ip distribution
        # power meter mean photon number/ or maybe the 100kHz mean photon for that power
        # ref_mu = log_df.loc[(log_df['power_group'] == f'power_{power}') & (log_df['rep_rate/kHz'] == rep_rate), 'pm_estimated_av_pn'].iloc[0]
        # ref_mu = log_df.loc[(log_df['power_group'] == f'power_{power}') & (log_df['rep_rate/kHz'] == 100), 'ip_classifier_av_pn'].iloc[0]
        if rep_rate == 100:
            ref_mu = np.sum(ip_distrib * ip_labels)

        # get RF distribution
        rf_distrib = np.array(df_rf.loc[df_rf['rep_rate'] == int(rep_rate), '0':].iloc[0])
        rf_distrib = np.nan_to_num(rf_distrib)
        rf_labels = np.arange(len(rf_distrib))

        # calculate tvd
        tvds[0, i_rep] = tvd(ip_distrib, poisson_norm(ip_labels, ref_mu))
        tvds[1, i_rep] = tvd(rf_distrib, poisson_norm(rf_labels, ref_mu))

        # special power
        if power == special_power:
            special_distribs['ref'] = poisson_norm(ip_labels, ref_mu)
            if rep_rate == 100:
                special_distribs['ip_100'] = ip_distrib
            if rep_rate in special_rep_rates:
                special_distribs[f'rf_{rep_rate}'] = rf_distrib

    ref_mean_photons[power] = ref_mu

    # Plot TVD
    ax1.plot(rep_rates, tvds[0], 'o--', alpha=0.5, markersize=6, color=color)
    ax1.plot(rep_rates, tvds[1], 'o-', label=f'{ref_mu:.2f}', alpha=0.8, markersize=6, color=color)

    # highlight the special rep rate
    if power == special_power:
        id_100 = np.argmax(rep_rates == 100)
        for i in range(len(special_rep_rates)):
            id_special = np.argmax(rep_rates == special_rep_rates[i])
            ax1.plot(rep_rates[id_special], tvds[1, id_special], 'D', color=special_colors[i], zorder=4, markersize=8)

ax1.legend(loc='upper left', fontsize=fontsize - 2)
ax1.set_xlabel('Repetition rate/kHz', fontsize=fontsize)
ax1.set_ylabel('TVD', fontsize=fontsize)
ax1.set_xticks(rep_rates)
ax1.tick_params(labelsize=fontsize - 2)


'''Plot special '''
width=0.3

ip_100 = special_distribs['ip_100']
ref_poisson = special_distribs['ref']

ax2.bar(np.arange(len(ip_100)) - width, ip_100, width=width, align='center', alpha=0.8, label='100kHz', color='black')
for i_special, special_rep_rate in enumerate(special_rep_rates):
    rf_high = special_distribs[f'rf_{special_rep_rate}']
    ax2.bar(np.arange(len(rf_high)) + width * i_special, rf_high, width=width, align='center', alpha=0.8,
            label=f'{special_rep_rate}kHz', color=special_colors[i_special])
ax2.plot(np.arange(len(ip_100)), ref_poisson, 'x--', color='red', alpha=0.5)

ax2.set_ylim(bottom=0)
ax2.set_ylabel('Probability', fontsize=fontsize)
ax2.set_xlabel('Photon number', fontsize=fontsize)
ax2.tick_params(labelsize=fontsize-2)
ax2.set_xlim(-1, 15)
ax2.legend(loc='upper right', fontsize=fontsize-2)

# # fig2, ax2 = plt.subplots(figsize=(12, 6))
#
# df = results_dict[special_power]
# row_100 = df.loc[df['rep_rate']==100, :]
#
# pn_100= np.array(row_100.loc[:, '0':].iloc[0])
# pn = np.array(df.loc[df['rep_rate']== special_reprate, '0':].iloc[0])
# pn2 = np.array(df.loc[df['rep_rate']== special_reprate2, '0':].iloc[0])
# labels = np.arange(len(pn_100))
# width=0.3
#
# ax2.bar(labels - width, pn_100, width=width, align='center', alpha=0.8, label='100kHz', color='black')
# ax2.bar(labels, pn, width=width, align='center', alpha=0.8, label=f'{special_reprate}kHz', color='gray')
# ax2.bar(labels + width, pn2, width=width, align='center', alpha=0.8, label=f'{special_reprate2}kHz', color='saddlebrown')
#
# ax2.set_ylim(0, 0.5)
# ax2.set_ylabel('Probability', fontsize=fontsize)
# ax2.set_xlabel('Photon number', fontsize=fontsize)
# ax2.tick_params(labelsize=fontsize-2)
#
# # def poisson_norm(x, mu):
# #     return (mu ** x) * np.exp(-mu) / factorial(x)
#
# ax2.legend(loc='upper right', fontsize=fontsize-2)


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

plt.show()

