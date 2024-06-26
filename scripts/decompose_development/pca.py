import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.data_reader import DataReader
from tes_resolver.data_chopper import DataChopper
from tes_resolver.traces import Traces, TraceUtils
from tes_resolver.classifier import InnerProductClassifier

# Read data
dataReader = DataReader('Data/Tomography_data_2024_04')
data_group = 'power_6'
rep_rate = 800

data_raw = dataReader.read_raw_data(data_group, rep_rate)
trigger_delay = DataChopper.find_trigger(data_raw, samples_per_trace=int(5e4 / rep_rate))
targetTraces = Traces(rep_rate, data_raw, parse_data=True, trigger_delay=trigger_delay)

# use inner product classifier
ipClassifier = InnerProductClassifier()
ipClassifier.train(targetTraces)
ipClassifier.predict(targetTraces, update=True)

# get data from the first traces
traces_to_plot = 1000
data = targetTraces.data[:traces_to_plot]

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

# Truncate at first few principal components
pca_components = 5
F_truncated = F[:, :pca_components]
data_cleaned = F_truncated @ QT[:pca_components, :] + col_means

# plot components
plt.figure('Principle components')
for i in range(pca_components):
    plt.plot(QT[i], label=f'{i}')
plt.legend()

# plot factor scores
plt.figure('factor scores')
plt.plot(F[:, 0], F[:, 1], '.', ls='None', alpha=0.5)
plt.xlabel('0-th component factor')
plt.ylabel('1-st component factor')

ax = plt.figure('3d factor scores').add_subplot(projection='3d')
ax.plot(F[:, 0], F[:, 1], F[:, 2], '.', ls='None')
# for pn in range(max_pn+1):
#     F_pn = F[pn*100: (pn+1)*100]
#     ax.plot(F_pn[:, 0], F_pn[:, 1], F_pn[:, 2], '.', ls='None', label=f'{pn}')
ax.legend()

