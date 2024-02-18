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
results_dict = {}
for power in powers:
    df = pd.read_csv(rf'..\params\RF_results_raw_{power}.csv')
    df = df.sort_values('rep_rate')
    results_dict[power] = df


fig, ax1 = plt.subplots(figsize=(12, 6))
left, bottom, width, height = [0.2, 0.4, 0.4, 0.45]
ax2 = fig.add_axes([left, bottom, width, height])

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

    ax1.plot(freqs, tvds, 'o-', label=f'{mean_photon:.2f}', alpha=0.8)

    if power == special_power:
        id_100 = np.argmax(freqs==100)
        id_special = np.argmax(freqs==special_reprate)
        ax1.plot(freqs[id_100], tvds[id_100], 'D', color='black', zorder=4, markersize=8)
        ax1.plot(freqs[id_special], tvds[id_special], 'D', color='gray', zorder=4, markersize=8)

ax1.legend(loc='lower right')
ax1.set_xlabel('Repetition rate/kHz')
ax1.set_ylabel('TVD')

df = results_dict[special_power]
row_100 = df.loc[df['rep_rate']==100, :]

mu_100 = row_100.loc[0, 'fit_mu']
pn_100= np.array(row_100.loc[:, '0':].iloc[0])
pn = np.array(df.loc[df['rep_rate']== special_reprate, '0':].iloc[0])
labels = np.arange(len(pn_100))
width=0.4

ax2.bar(labels - width/2, pn_100, width=width, align='center', alpha=0.8, label='100kHz', color='black')
ax2.bar(labels + width/2, pn, width=width, align='center', alpha=0.8, label=f'{special_reprate}kHz', color='gray')
ax2.set_ylim(0, 0.25)
ax2.set_ylabel('Probability')
ax2.set_xlabel('Photon number')

def poisson_norm(x, mu):
    return (mu ** x) * np.exp(-mu) / factorial(x)
ax2.plot(labels, poisson_norm(labels, mu_100), 'rx-', label=rf'$\mu=${mu_100:.2f}')

ax2.legend(loc='upper right')
plt.show()

fig.savefig(DFUtils.create_filename(r'..\..\Plots\TVD_plots\TVD_plot.pdf'))