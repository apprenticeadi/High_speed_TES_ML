import numpy as np
import matplotlib.pyplot as plt

from src.utils import DataUtils
from src.traces import Traces

num_traces = 5

data_100 = DataUtils.read_raw_data(100)
train_100 = data_100[:num_traces, :].flatten()

plt.figure(f'100kHz raw trace train')
plt.plot(train_100)
plt.savefig(r'..\Plots\100kHz_5_trace_train.pdf')

# col_mean = np.mean(data_100, axis=0)
# data_100_zeroed = data_100 - col_mean
# train_100_zeroed = data_100_zeroed[:num_traces, :].flatten()
#
# plt.figure(f'100kHz zeroed trace train')
# plt.plot(train_100_zeroed)
#


frequency = 500
period = int(5e4 / frequency)
data_high = DataUtils.read_high_freq_data(frequency)
train_high = data_high[:num_traces, :].flatten()
plt.figure(f'{frequency}kHz trace train')
plt.plot(train_high)
plt.savefig(r'..\Plots\500kHz_5_trace_train.pdf')


train_100_overlapped = np.zeros(period * (num_traces-1) + 500)
for i in range(num_traces):
    train_100_overlapped[i * period: i * period + 500] += data_100[i, :]

plt.figure(f'Overlapped 100kHz trace train')
plt.plot(train_100_overlapped)
plt.savefig(r'..\Plots\100kHz_overlap_to_500kHz_5_trace_train.pdf')