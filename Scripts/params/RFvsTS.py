import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def create_color_plot(data, title, figsize=(8, 6)):
    # Define a custom colormap with more gradual transitions
    colors = [(0.0, 'green'), (0.5, 'yellow'), (0.8, 'red'), (1.0, 'black')]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    min_value = 1e-4  # Adjust this value to control the colormap range for small values
    norm = mcolors.LogNorm(vmin=min_value, vmax=data.max())

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create the grid of squares with the custom colormap and normalization
    cax = ax.matshow(data, cmap=cmap, norm=norm)

    # Add a colorbar
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('Ln(chi-square value)')

    # Set axis labels and ticks
    ax.set_xlabel('Reprate')
    ax.set_ylabel('Average Photon number')

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(['200', '300', '400', '500', '600', '700', '800', '900', '1000'])
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(['1.34', '3.3', '6.03', '8.14'])

    # Set the title
    plt.title(title)

    # Show the plot
    plt.show()

'''
load in chi square values computed for both tail subtraction and random forest with no FE
'''
data = np.loadtxt('TS_chi_vals.txt', unpack = True).T
data2 = np.loadtxt('RF_chi_vals.txt', unpack=True).T


create_color_plot(data, 'chi-square values for tail subtraction', figsize=(10,6))

create_color_plot(data2, 'chi-square values for Random forest', figsize=(10,6))










