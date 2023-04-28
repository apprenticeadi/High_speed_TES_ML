import numpy as np
import matplotlib.pyplot as plt
import os

from fitting_hist import fitting_histogram
from utils import read_high_freq_data, read_raw_data

# This script is built on Ruidi's original subtraction_analysis script
# %%

'''Calibration data'''
data_100_ = read_raw_data(100)
data_100 = data_100_.T
# %%

'''higher frequency data'''
frequency = 800
data_high = read_high_freq_data(frequency)
period = int(5e4 / frequency)

'''
Analysis for 100kHz, to find the characteristic shape of the waveform for each photon number
'''
min_voltage = np.amin(data_100)
max_voltage = np.amax(data_100)
ymin = 5000 * (min_voltage // 5000)
ymax = 5000 * (max_voltage // 5000 + 1)


'''
plot first _ traces in 100kHZ data
'''
n_traces_100 = 100

plt.figure(f'first {n_traces_100} 100kHz traces')
for i in range(n_traces_100):
    plt.plot(data_100[i][:200])
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.xlim(0, 200)
plt.ylim(ymin, ymax)

'''
plot average trace
'''
ave_trace = np.mean(data_100, axis=0)
#
offset = min(ave_trace)  # zero photons should give zero voltage.
data_100 -= offset

plt.figure('average trace 100kHz')
ave_trace = np.mean(data_100, axis=0)
plt.plot(ave_trace)
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.ylim(ymin, ymax)
plt.title('average trace')


'''
Plot first x traces of high rep rate data
'''
num_high_traces = 100
plt.figure(f'First {num_high_traces} traces of {frequency}kHz')
for i in range(num_high_traces):
    plt.plot(data_high[i])
plt.xlim(0, period)
# plt.show()
# # %%

'''
Plot average trace of high rep rate data
'''
plt.figure(f'average {frequency}kHz trace')
ave_trace_high = np.mean(data_high, axis=0)
plt.plot(ave_trace_high)
# plt.show()
# # %%

