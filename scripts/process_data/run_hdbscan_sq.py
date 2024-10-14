import numpy as np
import time
import os
import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

import hdbscan
from hdbscan import HDBSCAN

from tes_resolver import Traces, DataChopper, config, generate_training_traces
from tes_resolver.classifier import InnerProductClassifier, TabularClassifier, CNNClassifier
from utils import DFUtils, DataReader, RuquReader, LogUtils

modeltype = 'HDBSCAN'  # machine learning model

'''Run ml classifier to classify all the data in a certain folder. '''
sampling_rate = 5e4
data_name = r'squeezed states 2024_07_17'
dataReader = RuquReader(rf'Data\{data_name}')

# parameters
data_group = 'Chan[2]'
rep_rate = 800
data_time = '2024-07-17-2010'
num_training = 20000
num_predict = 30000

results_dir = os.path.join(config.home_dir, '..', 'Results', data_name, f'{modeltype}-{data_time}',
                               f'{data_group}_{rep_rate}kHz_{config.time_stamp}')

# logging
LogUtils.log_config(config.time_stamp, results_dir, 'log')
logging.info(f'Processing squeezed data with {rep_rate}kHz data collected on {data_time} with HDBSCAN.')

# Read data
data_raw = dataReader.read_raw_data(data_group, f'{rep_rate}kHz')
traces = Traces(rep_rate, data_raw, parse_data=True, trigger_delay='automatic')

# perform and plot PCA
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


# save f and qt
f_df = pd.DataFrame(f_data, columns=[f'F{i+1}' for i in range(f_data.shape[1])])

qt_df = pd.DataFrame(qt_data.T, columns=[f'PC{i+1}' for i in range(qt_data.shape[0])])
qt_df['time'] = np.arange(len(qt_data[0])) / sampling_rate * 1000
qt_df.to_csv(results_dir + rf'\principal_components.csv', index=False)

# plot f and qt
fig, axs = plt.subplots(3,1, layout='constrained', figsize=(8, 10))
ax = axs[0]
for i in range(len(qt_data)):
    ax.plot(np.arange(len(qt_data[i]))/ sampling_rate * 1000, qt_data[i], label=f'PC{i+1}')
ax.set_xlabel(r'$\mu s$')
ax.set_ylabel('arb. unit')
ax.legend()
ax.set_title('Principal components')


for i_data in range(2):
    ax = axs[i_data+1]

    if i_data == 0 :
        to_plot = f_data
    else:
        to_plot = f_rest

    x = to_plot[:, 0]
    y = to_plot[:, 1]
    heights, x_edges, y_edges = np.histogram2d(x, y, bins=(1000, 250), density=False)
    # interpolate to find the heights at (x,y) coordinates
    z = interpn((0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])),
                heights, np.vstack([x, y]).T, method="splinef2d", bounds_error=False, fill_value=1)
    z = z + (1 - np.min(z))
    # z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    density_df = pd.DataFrame(np.vstack([x, y, z]).T, columns=['F1', 'F2', 'interpn_num_points'])
    if i_data == 0:
        density_df.to_csv(results_dir + rf'\density_scatter_of_first_{num_training}.csv', index=False)
    else:
        density_df.to_csv(results_dir + rf'\density_scatter_of_rest_{len(f_rest)}.csv', index=False)

    # plot density scatter plot
    image = ax.scatter(x, y, c=z, s=5, cmap='viridis')
    cbar = fig.colorbar(image, ax=ax, location='bottom', aspect=30)
    cbar.ax.set_xlabel('Number of points')
    ax.set_xlabel(r'$F_1$')
    ax.set_ylabel(r'$F_2$')
    if i_data == 0:
        ax.set_title('Training')
    else:
        ax.set_title('Predict')

fig.savefig(results_dir + rf'\pca_results.pdf')

plt.show()
plt.pause(5)
plt.close('all')

'''Clustering'''
t0 = time.time()
cluster_model = HDBSCAN(min_cluster_size=20, min_samples=5, prediction_data=True, cluster_selection_epsilon=500).fit(f_data)
t_cluster = time.time() - t0  # time to cluster
print(f'Clustering took {t_cluster} seconds')

cluster_labels = cluster_model.labels_
all_cluster_labels = list(set(cluster_labels))
all_cluster_labels.sort()

f_df['cluster_label'] = cluster_labels
f_df.to_csv(results_dir + rf'\factor_scores_first_{num_training}_traces.csv', index=False)

color_cycle = list(plt.get_cmap('tab20').colors)
color_cycle = color_cycle[0::2] + color_cycle[1::2]

fig2, axs2 = plt.subplots(3, 1, layout='constrained', figsize=(8, 10))
ax = axs2[0]

medoids = np.zeros((len(all_cluster_labels)-1, 2))
for i in all_cluster_labels:
    indices = np.argwhere(cluster_labels == i).ravel()
    if i == -1:
        ax.scatter(f_data[indices, 0], f_data[indices, 1], alpha=0.5, label=f'{i}', s=0.5, color='black')
    else:
        # this is actually centroids- the mean of the cluster
        medoids[i] = (np.mean(f_data[indices, 0]), np.mean(f_data[indices, 1]))
        indices = indices[:200]
        ax.scatter(f_data[indices, 0], f_data[indices, 1], alpha=0.2, label=f'{i}', s=5,
                   color=color_cycle[i % len(color_cycle)])

ax.set_xlabel(r'$F_1$')
ax.set_ylabel(r'$F_2$')
ax.set_title('Training')

plt.show()
plt.pause(2)
fig2.savefig(results_dir + rf'\clustering_results.pdf')

'''Post process the clusters'''
medoid_df = pd.DataFrame(medoids, columns=['F1', 'F2'])
medoid_df['cluster_label'] = all_cluster_labels[1:]
medoid_df.to_csv(results_dir + rf'\centroids.csv', index=False)
sort_args = np.argsort(medoids[:,0])
sorted_medoids = medoids[sort_args]

df = pd.DataFrame(columns=['pn', 'cluster_label', 'medoid', 'num_training', 'training_prob',
                           'num_predict', 'predict_prob'])
df.loc[0, :'training_prob'] = [-1, -1, [], np.sum(cluster_labels == -1), np.sum(cluster_labels == -1) / len(cluster_labels)]

for idx, medoid in enumerate(sorted_medoids):
    cluster_label = np.argmax(medoids[:,0] == medoid[0])
    ax.plot(medoid[0], medoid[1], marker='x', alpha=1.0, color=color_cycle[cluster_label % len(color_cycle)],
            markersize=20)
    plt.pause(1)

    pn = int(input(f'Enter the photon number for medoid {medoid}:'))

    num_traces = np.sum(cluster_labels == cluster_label)
    df.loc[idx+1, :'training_prob'] = [pn, cluster_label, medoid, num_traces, num_traces / len(cluster_labels)]

'''predict the rest'''
t_1 = time.time()
predict_labels, strengths = hdbscan.approximate_predict(cluster_model, f_rest)
t_predict = time.time() - t_1

f_rest_df = pd.DataFrame(np.vstack([f_rest[:, 0], f_rest[:, 1], predict_labels, strengths]).T,
                         columns=['F1', 'F2', 'cluster_label', 'strength'])
f_rest_df.to_csv(results_dir + rf'\factor_scores_rest_{len(f_rest)}_traces.csv', index=False)

time_df = pd.DataFrame({'training_t': [t_cluster], 'training_traces':  [num_training],
                        'predict_t': [t_predict], 'predict_traces': [len(f_rest)]})
time_df.to_csv(results_dir + rf'\clustering_time.csv', index=False)

ax = axs2[1]
for label in df['cluster_label']:
    indices = np.argwhere(predict_labels == label).ravel()
    df.loc[df['cluster_label'] == label, 'num_predict'] = len(indices)
    df.loc[df['cluster_label'] == label, 'predict_prob'] = len(indices) / len(predict_labels)


    if label == -1:
        indices = indices[:1000]
        ax.scatter(f_rest[indices, 0], f_rest[indices, 1], alpha=0.5, label=f'{label}', s=0.5, color='black')
    else:
        indices = indices[:200]
        ax.scatter(f_rest[indices, 0], f_rest[indices, 1], alpha=0.2, label=f'{label}', s=5,
                   color=color_cycle[label % len(color_cycle)])

ax.set_xlim(axs2[0].get_xlim())
ax.set_ylim(axs2[0].get_ylim())
ax.set_xlabel(r'$F_1$')
ax.set_ylabel(r'$F_2$')
ax.set_title('Predict')
plt.pause(10)
df.to_csv(results_dir + rf'\clustered_pn_distribution.csv', index=False)

# Plot bar plot
ax = axs2[2]
total_traces = np.sum(df['num_training']) + np.sum(df['num_predict'])
bottoms = np.zeros(np.max(df['pn']) + 2)  # first one is for -1
for idx in range(len(df)):
    pn = df.loc[idx, 'pn']
    cluster_label = df.loc[idx, 'cluster_label']
    if cluster_label == -1:
        color='black'
    else:
        color= color_cycle[cluster_label % len(color_cycle)]

    ax.bar(pn, (df.loc[idx, 'num_training'] + df.loc[idx, 'num_predict'])/ total_traces, width=0.8, bottom=bottoms[pn+1],
           align='center', color=color, alpha=0.8)
    bottoms[pn+1] = bottoms[pn+1] + (df.loc[idx, 'num_training'] + df.loc[idx, 'num_predict']) / total_traces

ax.set_xlabel('Photon number')
ax.set_ylabel('Probability')

fig2.savefig(results_dir + rf'\clustering_results.pdf')


plt.pause(5)


