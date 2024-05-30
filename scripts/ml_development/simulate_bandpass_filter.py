import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal as signal

from tes_resolver.ml_funcs import generate_training_traces
from tes_resolver.traces import Traces
from tes_resolver.classifier import InnerProductClassifier
from tes_resolver.data_chopper import DataChopper
from src.data_reader import DataReader


dataReader = DataReader('Data/Tomography_data_2024_04')
data_group = 'power_10'

cal_rep_rate = 100
high_rep_rate = 1000
num_traces_to_plot = 100
start_plot = 1000
sampling_rate = 5e4  # kHz
filter_cutoff= 1000

# generate training traces (without any actual labels)
cal_data = dataReader.read_raw_data(data_group, cal_rep_rate)
calTraces = Traces(cal_rep_rate, cal_data, parse_data=True, trigger_delay=0)
calTraces.labels = np.zeros(calTraces.num_traces, dtype=int)  # give it fake labels

trainingTraces = generate_training_traces(calTraces, target_rep_rate=high_rep_rate, trigger_delay=0)

training_traces_data = trainingTraces.data.flatten()  # flatten the training traces data

fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True, sharey=True, layout='constrained')
ax1, ax2, ax3 = axs
ax1.plot(training_traces_data)
ax1.set_title('Training traces')
ax1.set_xlim(start_plot*trainingTraces.period, (start_plot + num_traces_to_plot) * trainingTraces.period)

# create filter
sos = signal.butter(100, filter_cutoff, btype='low', output='sos', fs=sampling_rate)
filtered = signal.sosfilt(sos, training_traces_data)

ax2.plot(filtered)
ax2.set_title('Filtered training traces')

# read actual data
actual_data = dataReader.read_raw_data(data_group, high_rep_rate)
actual_data = actual_data.flatten()

ax3.plot(actual_data[:len(training_traces_data)])
ax3.set_title('Actual data')

# plot filter
w, H = signal.sosfreqz(sos, fs=sampling_rate)
plt.figure('Filter')
plt.plot(w, 20*np.log10(np.maximum(1e-10, np.abs(H))))
plt.xlim(0, 2*filter_cutoff)
plt.show()