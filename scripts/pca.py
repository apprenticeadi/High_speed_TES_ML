import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils import DataReader
from tes_resolver import DataChopper, Traces

# Read data
dataReader = DataReader('Data/Tomography_data_2024_04')
data_group = 'power_9'
ref_rep_rate = 100
high_rep_rate = 800
sampling_rate = 5e4

# data_raw = dataReader.read_raw_data(data_group, ref_rep_rate)
# trigger_delay = 0 # DataChopper.find_trigger(data_raw, samples_per_trace=int(5e4 / rep_rate))
# calTraces = Traces(ref_rep_rate, data_raw, parse_data=True, trigger_delay=trigger_delay)
#
# # use inner product classifier
# ipClassifier = InnerProductClassifier()
# ipClassifier.train(calTraces)
# ipClassifier.predict(calTraces, update=True)
#
# # remove baseline
# cal_baseline = calTraces.find_offset()
# calTraces.data = calTraces.data - cal_baseline

# load actual data
actual_data = dataReader.read_raw_data(data_group, high_rep_rate)
trigger_delay = DataChopper.find_trigger(actual_data, samples_per_trace=int(sampling_rate/high_rep_rate))
targetTraces = Traces(high_rep_rate, actual_data, parse_data=True, trigger_delay=trigger_delay)

# # overlap to high rep rate
# targetTraces = generate_training_traces(calTraces, 800, trigger_delay)

# get data from the first traces
traces_to_plot = 1000
data = targetTraces.data

# # get data from each photon number bin
# max_pn = 5
# traces_to_plot = (max_pn+1)*100
# data = np.zeros((traces_to_plot, targetTraces.period))
# for pn in range(max_pn+1):
#     data[pn*100: (pn+1)*100] = targetTraces.pn_traces(pn)[:100]

# To perform PCA, first zero the mean along each column
col_means = np.mean(data, axis=0)
data_zeroed = data - col_means

plt.figure('zeroed data')
for i in range(traces_to_plot):
    plt.plot(data_zeroed[i], alpha=0.1)


# Singular value decomposition to find factor scores and loading matrix
P, Delta, QT = np.linalg.svd(data_zeroed, full_matrices=False)
F = P * Delta  # Factor scores

# # Truncate at first few principal components
# pca_components = 5
# F_truncated = F[:, :pca_components]
# data_cleaned = F_truncated @ QT[:pca_components, :] + col_means

# plot components
plt.figure('Principle components')
for i in range(5):
    plt.plot(QT[i], label=f'{i}')
plt.legend()

# # plot factor scores
# plt.figure('factor scores')
# # for pn in range(max_pn+1):
# #     F_pn = F[pn*100: (pn+1)*100]
# #     plt.plot(F_pn[:, 0], F_pn[:, 1], '.', ls='None', label=f'{pn}')
# plt.plot(F[:, 0], F[:, 1], '.', ls='None', alpha=0.5)
# plt.xlabel('0-th component factor')
# plt.ylabel('1-st component factor')
heatmap, xedges, yedges = np.histogram2d(F[:, 0], F[:, 1], bins=(512, 384))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.figure('Heat map of factor scores')
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()

# ax = plt.figure('3d factor scores 234').add_subplot(projection='3d')
# # ax.plot(F[:, 0], F[:, 1], F[:, 2], '.', ls='None')
# for pn in range(max_pn+1):
#     F_pn = F[pn*100: (pn+1)*100]
#     ax.plot(F_pn[:, 0], F_pn[:, 1], F_pn[:, 2], '.', ls='None', label=f'{pn}')
# ax.legend()
#
