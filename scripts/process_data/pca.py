import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize
import threading
import datetime
import os

from utils import DataReader, RuquReader, DFUtils
from tes_resolver import DataChopper, generate_training_traces
from tes_resolver.classifier import InnerProductClassifier, TabularClassifier
from tes_resolver.traces import Traces, PCATraces

from sklearn.cluster import AgglomerativeClustering, KMeans, Birch, DBSCAN, SpectralClustering, OPTICS, HDBSCAN
from sklearn.mixture import GaussianMixture

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


# Parameters
cal_rep_rate = 100
cal_keywords = ['2nmPump', '112uW', '2024-07-17-1954_']
caltraces = []  # calibration traces
trainingtraces = []  # training traces, from overlapping caltraces

high_rep_rate = 800
high_keywords = ['2nmPump', '900uW', '2024-07-17-2010_']
hightraces = []  # high rep rate traces

channels = ['Chan[1]', 'Chan[2]']

sampling_rate = 5e4
chop=True
trace_to_plot = 1000

sqReader = RuquReader(r'Data\squeezed states 2024_07_17')
ipClassifier = InnerProductClassifier(num_bins=1000, multiplier=1.)
mlClassifier = TabularClassifier('SVM', test_size=0.1)

# read raw data and process with inner product
for i_ch, channel in enumerate(channels):
    # read calibration data
    cal_data = sqReader.read_raw_data(f'{cal_rep_rate}kHz', channel, *cal_keywords, concatenate=True, return_file_names=False)
    if chop:
        cal_data = cal_data[:, :int(cal_data.shape[1]/2)]
        effective_rep_rate = 2*cal_rep_rate
    else:
        effective_rep_rate = cal_rep_rate
    calTraces = Traces(effective_rep_rate, cal_data, parse_data=False)
    # predict with inner product
    ipClassifier.train(calTraces)
    ipClassifier.predict(calTraces, update=True)
    # remove baseline
    baseline = calTraces.find_offset()
    calTraces.data = calTraces.data - baseline
    caltraces.append(calTraces)

    # read high rep rate data
    high_data = sqReader.read_raw_data(f'{high_rep_rate}kHz', channel, *high_keywords, concatenate=False, return_file_names=False)[0]
    highTraces = Traces(high_rep_rate, high_data, parse_data=True, trigger_delay='automatic')
    trigger_delay = highTraces.trigger_delay
    hightraces.append(highTraces)

    # generate training traces
    trainingTraces = generate_training_traces(calTraces, 800, trigger_delay=trigger_delay)
    # adjust vertical offset
    offset = np.max(trainingTraces.average_trace()) - np.max(highTraces.average_trace())
    trainingTraces.data = trainingTraces.data - offset

    trainingtraces.append(trainingTraces)

# find principal components of high rep rate data
qts = []
high_pcatraces = []
training_pcatraces = []
fig, axs = plt.subplots(3, 2, layout='constrained', figsize=(12, 8), sharex='all', sharey='row')  # raw traces and principle components
fig2 = plt.figure(layout='constrained', figsize=(12, 6))  # factor scores
for i_ch, channel in enumerate(channels):
    # high rep rate data
    high_data = hightraces[i_ch].data
    high_zeroed = high_data - np.mean(high_data, axis=0)
    # Perform PCA
    # Singular value decomposition to find factor scores and loading matrix
    P, Delta, QT = np.linalg.svd(high_zeroed, full_matrices=False)
    F_high = P * Delta  # Factor scores

    high_pcatraces.append(PCATraces(high_rep_rate, F_high[:, :5], labels=None))  # PCA trace with the first 5 pincipal components
    qts.append(QT)

    # training data
    training_data = trainingtraces[i_ch].data
    training_zeroed = training_data - np.mean(training_data, axis=0)
    # Perform PCA with existing qt
    QT_inv = np.linalg.inv(QT)
    F_training = training_zeroed @ QT_inv
    training_pcatraces.append(PCATraces(high_rep_rate, F_training[:, :5], labels=trainingtraces[i_ch].labels))
    #
    # # plot traces and principal components
    # ax0 = axs[0, i_ch]
    # for i in range(trace_to_plot):
    #     ax0.plot(high_zeroed[i], alpha=0.05)
    # ax0.set_title(f'{channel} zeroed {high_rep_rate}kHz traces')
    # ylims = [np.min(high_zeroed), np.max(high_zeroed)]
    # ax0.set_ylim(ylims)
    #
    # ax1 = axs[1, i_ch]
    # for i in range(trace_to_plot):
    #     ax1.plot(training_zeroed[i], alpha=0.05)
    # ax1.set_title(f'{channel} zeroed training traces')
    # ax1.set_ylim(ylims)
    #
    # ax2 = axs[2, i_ch]
    # for i in range(5):
    #     ax2.plot(QT[i], label=f'{i}', alpha=0.7)
    # ax2.set_title(f'{channel} principal components')
    # ax2.set_xlabel('Samples')
    # ax2.legend()

    # plot factor scores
    ax3 = fig2.add_subplot(2, 2, 1+i_ch, projection='scatter_density')
    density = ax3.scatter_density(F_high[:, 0], F_high[:, 1], cmap=white_viridis)
    ax3.set_title(f'{channel} {high_rep_rate}kHz factor scores')
    ax3.set_xlabel(r'$F_1$')
    ax3.set_ylabel(r'$F_2$')

    xlims = [np.min(F_high[:, 0]), np.max(F_high[:, 0])]
    ylims = [np.min(F_high[:, 1]), np.max(F_high[:, 1])]
    ax3.set_xlim(xlims)
    ax3.set_ylim(ylims)

    ax4 = fig2.add_subplot(2, 2, 3+i_ch, projection='scatter_density')
    density = ax4.scatter_density(F_training[:, 0], F_training[:, 1], cmap=white_viridis)
    ax4.set_title(f'{channel} training factor scores')
    ax4.set_xlabel(r'$F_1$')
    ax4.set_ylabel(r'$F_2$')
    ax4.set_xlim(xlims)
    ax4.set_ylim(ylims)

fig.suptitle(f'Traces and principal components')

fig2.colorbar(density, ax=[ax3, ax4], label='Number of points per pixel')
fig2.suptitle('Principal component factor scores')

# # run supervised tabular classifier
# modeltype = 'KNN'
#
# for i_ch, channel in enumerate(channels):
#     print(f'Training tabular classifier {modeltype} for {channel}...')
#     t0 = time.time()
#     mlClassifier = TabularClassifier(modeltype, test_size=0.1)
#     mlClassifier.train(training_pcatraces[i_ch])
#     print(f'Training finished after {time.time()-t0}s')
#
#     t1 = time.time()
#     mlClassifier.predict(high_pcatraces[i_ch], update=True)
#     print(f'Prediction finished after {time.time()-t1}s')
# #
# # # plot results
# fig3 = plt.figure(layout='constrained', figsize=(10, 16))  # plot first two factor scores for up to 5 photon number
# fig4, axs4 = plt.subplots(1,2, layout='constrained', figsize=(12, 6), sharex='all', sharey='all')  # plot pn distribution
# for i_ch, channel in enumerate(channels):
#     pcaTraces = high_pcatraces[i_ch]
#     indices_dict = pcaTraces.bin_traces()
#     ax = fig3.add_subplot(7, 2, i_ch+1, projection='scatter_density')
#     ax.scatter_density(pcaTraces.data[:, 0], pcaTraces.data[:, 1], cmap=white_viridis)
#     xlims = [np.min(pcaTraces.data[:, 0]), np.max(pcaTraces.data[:, 0])]
#     ylims = [np.min(pcaTraces.data[:, 1]), np.max(pcaTraces.data[:, 1])]
#     ax.set_xlim(xlims)
#     ax.set_ylim(ylims)
#     ax.set_title(f'{channel} factor scores')
#     if i_ch == 0:
#         ax.set_ylabel(r'$F_2$')
#
#     for pn in range(6):
#         ax = fig3.add_subplot(7, 2, 2*pn+3+i_ch, projection='scatter_density')
#         ax.scatter_density(pcaTraces.data[indices_dict[pn]][:, 0], pcaTraces.data[indices_dict[pn]][:, 1], cmap=white_viridis)
#         ax.set_title(f'factor scores for {pn} photons')
#         ax.set_xlim(xlims)
#         ax.set_ylim(ylims)
#         if pn == 5:
#             ax.set_xlabel(r'$F_1$')
#         if i_ch == 0:
#             ax.set_ylabel(r'$F_2$')
#
#     # plot pn distribution, compare with caltraces
#     ax = axs4[i_ch]
#     cal_labels, cal_distrib = caltraces[i_ch].pn_distribution(normalised=True)
#     pn_labels, distrib = pcaTraces.pn_distribution(normalised=True)
#     ax.bar(cal_labels-0.2, cal_distrib, width=0.4, label=f'{cal_rep_rate}kHz')
#     ax.bar(pn_labels+0.2, distrib, width=0.4, label=f'{high_rep_rate}kHz')
#     ax.legend()
#     ax.set_title(f'{channel} PN distribution')
#     ax.set_xlabel('Photon number')
#     ax.set_ylabel('Probability')
#
# fig3.suptitle(f'Scatter density plot for {high_rep_rate}kHz actual traces')

# clustering
f_data = high_pcatraces[0].data[:10000, :2]
fig5 = plt.figure('f data', layout='constrained')
ax = fig5.add_subplot(111, projection='scatter_density')
density = ax.scatter_density(f_data[:, 0], f_data[:, 1], cmap=white_viridis)
ax.set_title(f'{len(f_data)} factor scores')
ax.set_xlabel(r'$F_1$')
ax.set_ylabel(r'$F_2$')
fig5.colorbar(density, label='Number of points per pixel')

guess_clusters = 14
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# cache_dir = rf'..\..\Results\cache\OPTICS_{timestamp}'
# os.makedirs(cache_dir, exist_ok=True)


t1 = time.time()
# cluster_results = KMeans(n_clusters=guess_clusters).fit(f_data)
# cluster_model = AgglomerativeClustering(n_clusters=guess_clusters).fit(f_data)
# cluster_model = Birch(threshold=0.8, n_clusters=guess_clusters).fit(f_data)
# cluster_model = GaussianMixture(n_components=guess_clusters).fit(f_data)
# cluster_model = OPTICS(min_samples=100, memory=cache_dir).fit(f_data)
cluster_model = HDBSCAN(min_cluster_size=50, min_samples=5, cluster_selection_epsilon=700, store_centers='medoid').fit(f_data)
print(f'Time to cluster {len(f_data)} traces is {time.time()-t1}s')

t2 = time.time()
# cluster_labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(f_data[:10000])
# cluster_labels = SpectralClustering(n_clusters=guess_clusters).fit_predict(f_data[:10000])
# cluster_labels = cluster_model.predict(f_data)
cluster_labels = cluster_model.labels_
print(f'Time to predict {len(f_data)} traces is {time.time()-t2}s. The labels are {set(cluster_labels)}')

medoids = cluster_model.medoids_
clustered_dict = {}
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig6, axs6 = plt.subplots(1, 2, layout='constrained', figsize=(12, 6), sharex='all', sharey='all')
for i in set(cluster_labels):
    indices = np.argwhere(cluster_labels == i).ravel()

    ax = axs6[0]
    if i != -1:
        ax.plot(f_data[indices[:500], 0], f_data[indices[:500], 1], '.', ls='None', alpha=0.1,
                label=f'{i}', color=color_cycle[i % len(color_cycle)])

        ax.plot(medoids[i, 0], medoids[i, 1], 'x', color=color_cycle[i % len(color_cycle)], alpha=1.0, markersize=10)

    ax.set_title(r'pn$\geq0$')
    ax.set_xlabel(r'$F_1$')
    ax.set_ylabel(r'$F_2$')
    ax.legend()
    clustered_dict[i] = f_data[indices]

ax = axs6[1]
undecided = clustered_dict[-1]
ax.plot(undecided[:, 0], undecided[:, 1], '.', ls='None', alpha=0.1, label=f'{len(undecided)} undecided')
ax.legend()
ax.set_title(r'pn$=-1$')
ax.set_xlabel(r'$F_1$')
fig6.suptitle(f'Clustered factor scores for {len(f_data)} traces')

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



