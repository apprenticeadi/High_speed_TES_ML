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

raw5_rf = [0.0009461921378315901, 0.001480560963707156, 0.001726906595370491, 0.0010854924772207457,
      0.0021309504859532502, 0.0010354562250922748, 0.0009686258155935959, 0.002957139964012231, 0.007553264522824858]

raw6_rf = [0.06748699001773888, 0.05991548648299498, 0.16969965751650112, 0.40244329220324015,
           0.07935386294008585, 0.3569919375200459, 2.7363165504138482, 8.277931823526929, 12.597092128090116]
raw7_rf = [0.0008524608896246881, 0.0008400725045519428, 0.002324503514215332, 0.034372880841953585,
           0.01488382739422477, 0.0626440819682149, 0.36532730639629357, 0.9616255560424549, 1.9241376519281992]
raw8_rf = [0.00033063300749672984, 0.0013299690642947144, 0.0014392494787532823, 0.0018441796115673428,
           0.007241199225174884, 0.002927148476257993, 0.024696843756243498, 0.08082550368247564, 0.17130101069474554]

raw5_ts = [0.001,0.0014578857138759136, 0.002198602245027542, 0.0008324884954429709, 0.0026660928355385855,
           0.5732732979597663, 0.8069823521688169,50,50]
raw6_ts = [0.07,0.10197028290303171, 50, 50, 50, 50, 50,50,50]
raw7_ts = [0.004,0.0053730615623809475, 0.718652760833019, 54.74153946073283, 9.888020644372787, 50, 50,50,50]
raw8_ts = [0.001,0.0009915994883192328, 0.0026181772964979366, 0.25902425398586304, 0.0847931228583163, 0.07053938545421931, 50,50,50]


data = np.array([raw5_ts, raw8_ts, raw7_ts, raw6_ts])
data2 = np.array([raw5_rf, raw8_rf, raw7_rf, raw6_rf])

# Create subplots
figure, axis = plt.subplots(2, 1)


axis[0] = create_color_plot(data, 'chi-square values for tail subtraction', figsize=(10,6))

# Create the second subplot

axis[1] = create_color_plot(data2, 'chi-square values for Random forest', figsize=(10,6))

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()









