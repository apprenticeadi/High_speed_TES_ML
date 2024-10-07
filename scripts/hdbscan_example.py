import time
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import os
from scipy.interpolate import interpn
import pandas as pd
import random

import hdbscan
from hdbscan import HDBSCAN

from tes_resolver.traces import Traces
from utils import DataReader, RuquReader

modeltype='HDBSCAN'
num_training = 20000
num_predict = 30000

'''Read data'''
dataReader = RuquReader(r'Data/squeezed states 2024_07_17')
rep_rate = 800
data_time = '2024-07-17-2010'
data_keywords = [f'{rep_rate}kHz', data_time, '2nmPump', '1570nmBPF', 'Chan[1]']
data_raw = dataReader.read_raw_data(*data_keywords, concatenate=True)

traces = Traces(rep_rate, data_raw, parse_data=True, trigger_delay='automatic')

'''Perform and plot PCA'''
F, QT = traces.pca()
qt_data = QT[:2, :]  # first two principal components
# make qt positive
for i in range(2):
    if rep_rate == 100 and i == 1:
        pass
    else:
        sign = np.sign(max(qt_data[i], key=abs))
        qt_data[i] = qt_data[i] * sign
        F[:, i] = F[:, i] * sign

# first two factor scores for each trace
f_data = F[:num_training, :2]  # training
f_rest = F[num_training:num_training+num_predict, :2]  # predict

# plot factor scores
x = f_data[:, 0]
y = f_data[:, 1]
heights, x_edges, y_edges = np.histogram2d(x, y, bins=(1000, 250), density=False)
# interpolate to find the heights at (x,y) coordinates
z = interpn((0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])),
            heights, np.vstack([x, y]).T, method="splinef2d", bounds_error=False, fill_value=1)
z = z + (1 - np.min(z))
# z[np.where(np.isnan(z))] = 0.0
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
image = ax.scatter(x, y, c=z, s=5, cmap='viridis')
cbar = fig.colorbar(image, ax=ax, location='bottom', aspect=30)
cbar.ax.set_xlabel('Number of points')
ax.set_xlabel(r'$F_1$')
ax.set_ylabel(r'$F_2$')
ax.set_title('PCA result')

'''Clustering'''
t0 = time.time()
cluster_model = HDBSCAN(min_cluster_size=20, min_samples=5, prediction_data=True, cluster_selection_epsilon=500).fit(f_data)
t_cluster = time.time() - t0  # time to cluster
print(f'Clustering {num_training} traces took {t_cluster} seconds')

cluster_labels = cluster_model.labels_
all_cluster_labels = list(set(cluster_labels))
all_cluster_labels.sort()

'''Plot clustering result'''
color_cycle = list(plt.get_cmap('tab20').colors)
color_cycle = color_cycle[0::2] + color_cycle[1::2]

medoids = np.zeros((len(all_cluster_labels)-1, 2))

fig2, ax2 = plt.subplots()
for i in all_cluster_labels:
    indices = np.argwhere(cluster_labels == i).ravel()
    if i == -1:
        # unclassified data
        ax2.scatter(f_data[indices, 0], f_data[indices, 1], alpha=0.5, label=f'{i}', s=0.5, color='black')
    else:
        # this is actually centroids- the mean of the cluster
        medoids[i] = (np.mean(f_data[indices, 0]), np.mean(f_data[indices, 1]))
        indices = indices[:200]
        ax2.scatter(f_data[indices, 0], f_data[indices, 1], alpha=0.2, label=f'{i}', s=5,
                   color=color_cycle[i % len(color_cycle)])
ax2.plot(medoids[:, 0], medoids[:, 1], 'X', color='black', label='Medoids')

ax2.set_xlabel(r'$F_1$')
ax2.set_ylabel(r'$F_2$')
ax2.set_title('Clustering training result')

'''predict the rest'''
t_1 = time.time()
predict_labels, strengths = hdbscan.approximate_predict(cluster_model, f_rest)
t_predict = time.time() - t_1
print(f'Approximate predict {num_predict} traces took {t_predict} seconds')

plt.show()


