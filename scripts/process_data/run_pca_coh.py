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
import logging

from utils import DataReader, RuquReader, DFUtils, LogUtils
from tes_resolver import DataChopper, Traces, config

from sklearn.cluster import HDBSCAN

modeltype='PCA'
time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")

'''Which data to use'''
dataReader = DataReader(r'\Data\Tomography_data_2024_04')
data_group = 'power_9'
rep_rate = 800
num_traces = 10000  # number of traces to process

results_dir = rf'..\..\Results\Tomography_data_2024_04\{modeltype}\{data_group}\{rep_rate}kHz_{time_stamp}'
os.makedirs(results_dir, exist_ok=True)

LogUtils.log_config(time_stamp, dir=results_dir)

'''Read data'''
data_raw = dataReader.read_raw_data(data_group, rep_rate=rep_rate)
tarTraces = Traces(rep_rate, data_raw, parse_data=True, trigger_delay='automatic')

'''Perform and plot PCA'''
F, QT = tarTraces.pca()  # factor scores and loading matrix (principal components)
f_data = F[:num_traces, :2]  # first two factor scores for each trace
qt_data = QT[:2, :]  # first two principal components
# make qt positive
for i in range(2):
    if rep_rate == 100 and i == 1:
        pass
    else:
        sign = np.sign(max(qt_data[i], key=abs))
        qt_data[i] = qt_data[i] * sign
        f_data[:, i] = f_data[:, i] * sign

# save f and qt
f_df = pd.DataFrame(f_data, columns=[f'F{i+1}' for i in range(f_data.shape[1])])
f_df.to_csv(results_dir + rf'\factor_scores.csv', index=False)

qt_df = pd.DataFrame(qt_data.T, columns=[f'PC{i+1}' for i in range(qt_data.shape[0])])
qt_df['time'] = np.arange(len(qt_data[0])) / tarTraces.sampling_rate * 1000
qt_df.to_csv(results_dir + rf'\principal_components.csv', index=False)

# plot f and qt
fig, axs = plt.subplots(2,1, layout='constrained', figsize=(8, 10))
ax = axs[0]
for i in range(len(qt_data)):
    ax.plot(np.arange(len(qt_data[i]))/ tarTraces.sampling_rate * 1000, qt_data[i], label=f'PC{i+1}')
ax.set_xlabel(r'$\mu s$')
ax.set_ylabel('arb. unit')
ax.legend()

ax = axs[1]
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

density_df = pd.DataFrame(np.vstack([x, y, z]).T, columns=['F1', 'F2', 'interpn_num_points'])
density_df.to_csv(results_dir + rf'\density_scatter_of_first_2_factor_scores.csv', index=False)

# plot density scatter plot
image = ax.scatter(x, y, c=z, s=5, cmap='viridis')
cbar = fig.colorbar(image, ax=ax, location='bottom', aspect=30)
cbar.ax.set_xlabel('Number of points')
ax.set_xlabel(r'$F_1$')
ax.set_ylabel(r'$F_2$')
fig.savefig(results_dir + rf'\pca_results.pdf')

plt.show()
plt.pause(5)
plt.close('all')

'''Clustering'''
logging.info('Clustering with HDBSCAN ...')
t0 = time.time()
cluster_model = HDBSCAN(min_cluster_size=50, min_samples=5, store_centers='medoid').fit(f_data)
t_cluster = time.time() - t0  # time to cluster
logging.info(f'Clustering took {t_cluster} seconds')

cluster_labels = cluster_model.labels_
medoids = cluster_model.medoids_

f_df['cluster_label'] = cluster_labels
f_df.to_csv(results_dir + rf'\factor_scores.csv', index=False)

medoids_df = pd.DataFrame(medoids, columns=[f'F{i+1}' for i in range(medoids.shape[1])])
medoids_df['cluster_label'] = np.arange(len(medoids))
medoids_df.to_csv(results_dir + rf'\medoids.csv', index=False)

color_cycle = list(plt.get_cmap('tab20').colors)
random.shuffle(color_cycle)
fig2, axs2 = plt.subplots(2, 1, layout='constrained', figsize=(8, 10))
ax = axs2[0]

for i in set(cluster_labels):
    indices = np.argwhere(cluster_labels == i).ravel()
    if i == -1:
        ax.scatter(f_data[indices, 0], f_data[indices, 1], alpha=0.5, label=f'{i}', s=0.5, color='black')
    else:
        indices = indices[:200]
        ax.scatter(f_data[indices, 0], f_data[indices, 1], alpha=0.2, label=f'{i}', s=5,
                   color=color_cycle[i % len(color_cycle)])

ax.set_xlabel(r'$F_1$')
ax.set_ylabel(r'$F_2$')


plt.show()
plt.pause(10)


'''Post process the clusters'''
sort_args = np.argsort(medoids[:,0])
sorted_medoids = medoids[sort_args]
all_cluster_labels = list(set(cluster_labels))
all_cluster_labels.sort()

df = pd.DataFrame(columns=['pn', 'cluster_label', 'medoid', 'num_traces', 'probability'])
df.loc[0] = [-1, -1, [], np.sum(cluster_labels == -1), np.sum(cluster_labels == -1) / len(cluster_labels)]

for idx, medoid in enumerate(sorted_medoids):
    cluster_label = np.argmax(medoids[:,0] == medoid[0])
    ax.plot(medoid[0], medoid[1], marker='x', alpha=1.0, color=color_cycle[cluster_label % len(color_cycle)],
            markersize=20)
    plt.pause(1)

    pn = int(input(f'Enter the photon number for medoid {medoid}:'))

    num_traces = np.sum(cluster_labels == cluster_label)
    df.loc[idx+1] = [pn, cluster_label, medoid, num_traces, num_traces / len(cluster_labels)]

df.to_csv(results_dir + rf'\clustered_pn_distribution.csv', index=False)

# Plot bar plot
ax = axs2[1]
total_traces = np.sum(df['num_traces'])
bottoms = np.zeros(np.max(df['pn']) + 2)  # first one is for -1
for idx in range(len(df)):
    pn = df.loc[idx, 'pn']
    if pn == -1:
        color='black'
    else:
        color= color_cycle[df.loc[idx, 'cluster_label'] % len(color_cycle)]

    ax.bar(pn, df.loc[idx, 'num_traces'] / total_traces, width=0.8, bottom=bottoms[pn+1],
           align='center', color=color, alpha=0.8)
    bottoms[pn+1] = bottoms[pn+1] + df.loc[idx, 'num_traces'] / total_traces

ax.set_xlabel('Photon number')
ax.set_ylabel('Probability')

'''Load calibration distribution'''
cal_dir = rf'..\..\Results\Tomography_data_2024_04\Params\IP'
cal_df = pd.read_csv(cal_dir + rf'\IP_results_{data_group}.csv')
cal_distrib = cal_df.loc[cal_df['rep_rate'] == 100, '0':].values[0]
n_error_df = pd.read_csv(cal_dir + rf'\bootstrapped\IP_results_{data_group}_n_error.csv')
n_errors = n_error_df.loc[n_error_df['rep_rate'] == 100, '0':].values[0]
p_errors_df = pd.read_csv(cal_dir + rf'\bootstrapped\IP_results_{data_group}_p_error.csv')
p_errors = p_errors_df.loc[p_errors_df['rep_rate'] == 100, '0':].values[0]

compare_df = pd.DataFrame(data=np.vstack((np.arange(len(cal_distrib)), cal_distrib, n_errors, p_errors)).T,
                          columns=['pn', 'probability', 'n_error', 'p_error'])

compare_df.to_csv(results_dir + rf'\calibration_pn_distribution_by_IP_100kHz.csv', index=False)

ax.errorbar(compare_df['pn'], compare_df['probability'], yerr=[compare_df['n_error'], compare_df['p_error']],
             marker='.', ls='--', label='Calibration', color='red')

largest_pn = np.argmax(cal_distrib==0) - 1
ax.set_xlim(-2, largest_pn+1)
ax.set_xticks(np.arange(largest_pn+2)-1)

plt.pause(5)

fig2.savefig(results_dir + rf'\clustering_results.pdf')
