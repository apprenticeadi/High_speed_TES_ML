import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.utils import DFUtils

sub_dir = r'\raw_[5, 8, 7]_2024-02-19(00-08-36.563517)'

results_dir = r'C:\Users\zl4821\PycharmProjects\TES_python\Results\Tomography' + sub_dir

to_plot = [100, 500, 700]
labels = ['(a)', '(b)', '(c)']

fig, axs = plt.subplots(1, 3, squeeze=True, sharey=True, layout='constrained', figsize=(12,3))

for id, ax in enumerate(axs):
    ax.set_title(f'{labels[id]} {to_plot[id]}kHz', fontfamily='serif', loc='left')

    estimated_theta = np.load(results_dir + rf'\{to_plot[id]}kHz_theta.npy')
    x = np.arange(estimated_theta.shape[1])
    y = np.arange(estimated_theta.shape[0])
    X, Y = np.meshgrid(x, y)

    pc = ax.pcolormesh(X, Y, estimated_theta, norm=mcolors.SymLogNorm(vmin=0, vmax=1, linthresh=0.01))

    ax.set_xticks(x[::2])
    ax.set_yticks(y[::2])

    if id==0:
        ax.set_ylabel(r'$n$')
    ax.set_xlabel(r'$m$')

cbar = fig.colorbar(pc, ax=axs.ravel().tolist())
cbar.set_label(r'$|\theta_{nm}|$')

plt.show()

fig.savefig(DFUtils.create_filename(fr'..\..\Plots\tomography_plots' + sub_dir + r'\tomography.pdf'))

optimal_values = np.load(results_dir + rf'\optimal_least_squares.npy')