import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize
import threading
import os

from utils import DataReader, RuquReader, DFUtils

from tes_resolver import DataChopper, Traces, config
from tes_resolver.classifier import InnerProductClassifier

from sklearn.cluster import AgglomerativeClustering, KMeans


def project_point_onto_line(m, x, y):
    """
    Project a point (x, y) onto a line defined by y = mx.
    Returns the distance from origin to the projected point.
    """
    return (x+m*y) / np.sqrt(m**2+1)


# Define a function to ask for user input
def get_user_input(prompt, result_list):
    result_list[0] = input(prompt)


# Read data
modeltype='PCA'
target_rep_rate = 800
sampling_rate = 5e4

chop=False
data_date = '2024-07-17-2010'
data_keywords = ['2nmPump', '900uW', data_date]
channels = ['Chan[1]', 'Chan[2]']
fig_titles = f'{target_rep_rate}kHz_' + '_'.join(data_keywords)

sqReader = RuquReader(r'Data\squeezed states 2024_07_17')
results_dir = os.path.join(config.home_dir, '..', 'Results', 'squeezed states 2024_07_17', modeltype,
                           f'{target_rep_rate}kHz_{data_date}_chop={chop}_{config.time_stamp}')

# plot scatter plot with density color
# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

fig = plt.figure(layout='constrained', figsize=(19, 9))
for i_ch, channel in enumerate(channels):
    print(f'\nProcessing {channel}...')
    data = sqReader.read_raw_data(f'{target_rep_rate}kHz', channel, *data_keywords, concatenate=True, return_file_names=False)

    if chop:
        data = data[:, :int(data.shape[1]/2)]
        effective_rep_rate = 2*target_rep_rate
    else:
        effective_rep_rate = target_rep_rate

    targetTraces = Traces(effective_rep_rate, data, parse_data=True)
    data = targetTraces.data

    # To perform PCA, first zero the mean along each column
    col_means = np.mean(data, axis=0)
    data_zeroed = data - col_means

    # Singular value decomposition to find factor scores and loading matrix
    P, Delta, QT = np.linalg.svd(data_zeroed, full_matrices=False)
    F = P * Delta  # Factor scores

    # save first two factor scores
    np.savetxt(DFUtils.create_filename(results_dir + rf'\{channel}_factor_scores.txt'), F[:, :2])

    ax1 = fig.add_subplot(3, 2, 1+i_ch, projection='scatter_density')
    density = ax1.scatter_density(F[:, 0], F[:, 1], cmap=white_viridis)
    fig.colorbar(density, ax=ax1, label='Number of points per pixel')
    ax1.set_xlabel(r'$F_1$')
    ax1.set_ylabel(r'$F_2$')
    plt.show(block=False)

    # ask input from user for the cut, plt.pause until user enters input
    # Create a list to store the user input
    user_input = [None]
    # Create a thread to ask for user input
    input_thread = threading.Thread(target=get_user_input, args=("Enter the gradient of the line to cut the data:", user_input))
    # Start the thread
    input_thread.start()
    # Display the plot and wait for the user input
    while input_thread.is_alive():
        plt.pause(1)
    # Get the user input from the list
    m_opt = float(user_input[0])

    # plot the cut and histogram
    xs = np.linspace(np.min(F[:, 0]), np.max(F[:, 0]), 100)
    ax1.plot(xs, m_opt * xs, 'r--', label=f'{m_opt:.2f}x')

    projected_F = [project_point_onto_line(m_opt, x, y) for x, y in F[:, :2]]
    ax2=fig.add_subplot(3, 2, 3+i_ch)
    heights, bin_edges, _ = ax2.hist(projected_F, bins=1000)
    heights = np.append(heights, 0)
    ax2.set_title('PCA stegosaurus')
    ax2.set_xlabel('Projected factor scores')
    ax2.set_ylabel('Occurrence')
    # ax2.invert_xaxis()

    # save projected F and histogram
    np.savetxt(DFUtils.create_filename(results_dir + rf'\{channel}_projected_F.txt'), projected_F)
    np.savetxt(DFUtils.create_filename(results_dir + rf'\{channel}_histogram.txt'), np.stack((bin_edges, heights)))

fig.suptitle(fig_titles)
fig.savefig(DFUtils.create_filename(results_dir + rf'\{modeltype}_stegosaurus.png'))







# # plot factor scores
# xrange = np.max(F[:, 0]) - np.min(F[:, 0])
# yrange = np.max(F[:, 1]) - np.min(F[:, 1])
# heatmap, xedges, yedges = np.histogram2d(F[:, 0], F[:, 1], bins=(int(xrange/300+0.5), int(yrange/300+0.5)))
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#
# plt.figure('Heat map of factor scores', layout='constrained')
# plt.imshow(heatmap.T, extent=extent, origin='lower', norm='linear')
# plt.xlabel(r'$F_1$')
# plt.ylabel(r'$F_2$')

# # find the optimal line to cut the data- doesn't work well.

#
#
# def calculate_histogram(m, points, num_bins=1000):
#     """
#     Calculate the histogram of projected points onto a line defined by y = mx.
#     Returns the histogram and the bin edges.
#     """
#     projected_points = [project_point_onto_line(m, x, y) for x, y in points]
#     hist, bin_edges = np.histogram(projected_points, bins=num_bins)
#     return hist, bin_edges
#
#
# def cost_function(m, points, num_bins=1000, threshold=50):
#     """
#     Calculate the cost of a line cut defined by y = mx + c.
#     The cost is the negative of the number of bins in the histogram that have more than 'threshold' points.
#     """
#     hist, _ = calculate_histogram(m, points, num_bins)
#     cost = -np.sum(hist > threshold)
#     return cost
#
#
# # Use an optimization algorithm to find the gradient and intercept that minimize the cost
# result = minimize(cost_function, 0.25, args=(F[:, :2], 500))
#
# # The optimal gradient and intercept
# m_opt = result.x[0]


# # clustering
# load ref data to know the number of clusters
# ref_data = dataReader.read_raw_data(data_group, 100)
# refTraces = Traces(100, ref_data, parse_data=True, trigger_delay=0)
# ipClassifier = InnerProductClassifier()
# ipClassifier.train(refTraces)
# ipClassifier.predict(refTraces, update=True)

# t1 = time.time()
# cluster_results = KMeans(n_clusters=len(set(refTraces.labels))).fit(f_data)
# t2 = time.time()
# print(f'Time to cluster {len(f_data)} traces is {t2-t1}s')
#
# cluster_labels = cluster_results.labels_
#
# clustered_dict = {}
# plt.figure('Clustered')
# for i in range(len(set(refTraces.labels))):
#     indices = np.argwhere(cluster_labels == i).ravel()
#
#     plt.plot(f_data[indices, 0], f_data[indices, 1], '.', ls='None', alpha=0.1, label=f'{i}')
#
#     clustered_dict[i] = data[:traces_to_plot][indices]
#
# # sort the labels by mean voltage value of trace
# mean_v = np.zeros(len(clustered_dict.keys()))
# for i in clustered_dict.keys():
#     if len(clustered_dict[i])>=1:
#         mean_v[i] = np.mean(clustered_dict[i])
#     else:
#         mean_v[i] = np.nan
#
# # get pn distribution predicted by pca clustering
# pn_distrib = np.zeros(len(mean_v))
# for pn, cl_label in enumerate(np.argsort(mean_v)):
#     pn_distrib[pn] = len(clustered_dict[cl_label])
# pn_distrib = pn_distrib / np.sum(pn_distrib)
#
# # plot pn distribution
# plt.figure('PN distribution')
# ref_pns, ref_distrib = refTraces.pn_distribution(normalised=True)
# plt.bar(ref_pns-0.2, ref_distrib, width=0.4, label='inner product')
# plt.bar(np.arange(len(pn_distrib))+0.2, pn_distrib, width=0.4, label='pca clustering')
#
# plt.show()

