import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
'''
script to create color plots for model comparison
'''
def create_color_plot(data, title, figsize=(8, 6)):

    colors = [(0.0, 'green'), (0.5, 'yellow'), (0.8, 'red'), (1.0, 'black')]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    min_value = 1e-4
    norm = mcolors.LogNorm(vmin=min_value, vmax=data.max())

    fig, ax = plt.subplots(figsize=figsize)

    cax = ax.matshow(data, cmap=cmap, norm=norm)

    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('Ln(chi-square value)')

    ax.set_xlabel('Reprate')
    ax.set_ylabel('Average Photon number')

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(['200', '300', '400', '500', '600', '700', '800', '900', '1000'])
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(['1.34', '3.3', '6.03', '8.14'])


    plt.title(title)


    plt.show()

'''
load in chi square values computed for both tail subtraction and random forest with no FE
'''
data = np.loadtxt('TS_chi_vals.txt', unpack = True).T
data2 = np.loadtxt('RF_chi_vals.txt', unpack=True).T


create_color_plot(data, 'chi-square values for tail subtraction', figsize=(10,6))

create_color_plot(data2, 'chi-square values for Random forest', figsize=(10,6))









