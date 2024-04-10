import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import factorial

from src.utils import DFUtils
def tvd(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return 0.5 * np.sum(np.absolute(a-b))

powers = [5,8,7]
special_power = 8
special_reprate = 700
special_reprate2 = 900
results_dict = {}
for power in powers:
    df = pd.read_csv(rf'..\..\Params\RF_results_raw_{power}.csv')
    df = df.sort_values('rep_rate')
    results_dict[power] = df

fontsize=14
fig, axs = plt.subplot_mosaic([['(a)', '(b)']], figsize=(12, 4), layout='constrained')
# left, bottom, width, height = [0.2, 0.4, 0.4, 0.45]
# ax2 = fig.add_axes([left, bottom, width, height])

# plt.subplots_adjust(wspace=0.3, hspace=0.1)

for label, ax in axs.items():
    ax.set_title(label, fontfamily='serif', loc='left', fontsize=fontsize+2)

ax1 = axs['(a)']
ax2 = axs['(b)']

for i, power in enumerate(powers):
    df = results_dict[power]

    freqs = np.array(df.loc[:, 'rep_rate'])
    tvds = np.zeros(len(freqs))
    pn_100 = np.array(df.loc[df['rep_rate']==100, '0':].iloc[0])
    labels = np.arange(len(pn_100))
    mean_photon = np.sum(pn_100 * labels)
    for i, freq in enumerate(freqs):
        pn = np.array(df.loc[df['rep_rate']== int(freq), '0':].iloc[0])
        tvds[i] = tvd(pn, pn_100)

    ax1.plot(freqs, tvds, 'o-', label=f'{mean_photon:.2f}', alpha=0.8, markersize=8)

    if power == special_power:
        id_100 = np.argmax(freqs==100)
        id_special = np.argmax(freqs==special_reprate)
        id_special2 = np.argmax(freqs==special_reprate2)
        ax1.plot(freqs[id_100], tvds[id_100], 'D', color='black', zorder=4, markersize=12)
        ax1.plot(freqs[id_special], tvds[id_special], 'D', color='gray', zorder=4, markersize=12)
        ax1.plot(freqs[id_special2], tvds[id_special2], 'D', color='saddlebrown', zorder=4, markersize=12)

ax1.legend(loc='upper left', fontsize=fontsize-2)
ax1.set_xlabel('Repetition rate/kHz', fontsize=fontsize)
ax1.set_ylabel('TVD', fontsize=fontsize)
ax1.tick_params(labelsize=fontsize-2)

df = results_dict[special_power]
row_100 = df.loc[df['rep_rate']==100, :]

mu_100 = row_100.loc[0, 'fit_mu']
pn_100= np.array(row_100.loc[:, '0':].iloc[0])
pn = np.array(df.loc[df['rep_rate']== special_reprate, '0':].iloc[0])
pn2 = np.array(df.loc[df['rep_rate']== special_reprate2, '0':].iloc[0])
labels = np.arange(len(pn_100))
width=0.3

ax2.bar(labels - width, pn_100, width=width, align='center', alpha=0.8, label='100kHz', color='black')
ax2.bar(labels, pn, width=width, align='center', alpha=0.8, label=f'{special_reprate}kHz', color='gray')
ax2.bar(labels + width, pn2, width=width, align='center', alpha=0.8, label=f'{special_reprate2}kHz', color='saddlebrown')

ax2.set_ylim(0, 0.3)
ax2.set_ylabel('Probability', fontsize=fontsize)
ax2.set_xlabel('Photon number', fontsize=fontsize)
ax2.tick_params(labelsize=fontsize-2)

def poisson_norm(x, mu):
    return (mu ** x) * np.exp(-mu) / factorial(x)
ax2.plot(labels - width, poisson_norm(labels, mu_100), 'rx-', label=rf'$\mu=${mu_100:.2f}')

ax2.legend(loc='upper right', fontsize=fontsize-2)
plt.show()

# fig.savefig(DFUtils.create_filename(r'..\..\Plots\TVD_plots\TVD_plot_sidebyside_3specials.pdf'))


plt.figure('accuracy')
for power in powers:
    df = results_dict[power]

    freqs = np.array(df.loc[:, 'rep_rate'])
    accuracies = np.array(df.loc[:, 'acc_score'])
    pn_100 = np.array(df.loc[df['rep_rate'] == 100, '0':].iloc[0])
    labels = np.arange(len(pn_100))
    mean_photon = np.sum(pn_100 * labels)

    plt.plot(freqs, accuracies, 'o-', label=f'{mean_photon:.2f}', alpha=0.8, markersize=8)

plt.legend(fontsize=fontsize-2)
plt.ylabel('Test accuracy score', fontsize=fontsize)
plt.xlabel('Repetition rate/kHz', fontsize=fontsize)
plt.tick_params(labelsize = fontsize-2)
# plt.savefig(r'..\..\Plots\TVD_plots\RF_accuracy_scores.pdf')
