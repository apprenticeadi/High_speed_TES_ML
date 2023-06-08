import numpy as np
import matplotlib.pyplot as plt

from src.traces import Traces
from src.utils import read_high_freq_data, read_raw_data

# A script that performs PCA on raw data traces. However, so far the improvement in fitting histogram is very limited.


frequency = 800

if frequency == 100:
    data_raw = read_raw_data(frequency)
else:
    data_raw = read_high_freq_data(frequency)

rawTraces = Traces(frequency=frequency, data=data_raw)



'''
Plot first x traces 
'''
num_traces = 20
offset, data_shifted = rawTraces.subtract_offset()
rawTraces.plot_traces(num_traces=num_traces, fig_name=f'first {num_traces} raw but shifted traces')
raw_hist_fit = rawTraces.fit_histogram(plot=True, fig_name='Raw histogram fit')
raw_char_traces = rawTraces.characteristic_traces_pn(plot=True, fig_name='raw char traces')

'''
To perform PCA, first zero the mean along each column
'''
col_means = np.mean(data_shifted, axis=0)
data_zeroed = data_shifted - col_means
zeroedTraces = Traces(frequency=frequency, data=data_zeroed)
zeroedTraces.plot_traces(num_traces=num_traces, fig_name=f'first {num_traces} column mean zeroed traces')

'''
Singular value decomposition to find factor scores and loading matrix
'''
P, Delta, QT = np.linalg.svd(data_zeroed, full_matrices=False)
F = P * Delta  # Factor scores
Q = QT.T  # Loading matrix

F2 = F ** 2
inertia_component = np.sum(F2, axis = 0)  # Sum of f squared for each principal component#
inertia_observation  = np.sum(F2, axis=1)  # Sum of all factor scores squared for each observation
total_inertia = np.sum(inertia_observation)
component_importance = inertia_component / total_inertia

'''
Truncate at first few principal components
'''
num_comp = 1
F_truncated = F[:, :num_comp]
data_cleaned = F_truncated @ QT[:num_comp, :] + col_means
cleanedTraces = Traces(frequency=frequency, data=data_cleaned)

cleanedTraces.plot_traces(num_traces=num_traces, fig_name=f'first {num_traces} PCA cleaned traces')
clean_hist_fit = cleanedTraces.fit_histogram(plot=True, fig_name='cleaned histogram fit')
clean_char_traces = cleanedTraces.characteristic_traces_pn(plot=True, fig_name='cleaned char traces')