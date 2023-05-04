import numpy as np
import matplotlib.pyplot as plt

from src.utils import read_high_freq_data, read_raw_data
from src.traces import Traces

num_traces = 10

data_100 = read_raw_data(100)
train_100 = data_100[:num_traces, :].flatten()

plt.figure(f'100kHz raw trace train')
plt.plot(train_100)

col_mean = np.mean(data_100, axis=0)
data_100_zeroed = data_100 - col_mean
train_100_zeroed = data_100_zeroed[:num_traces, :].flatten()

plt.figure(f'100kHz zeroed trace train')
plt.plot(train_100_zeroed)



frequency = 500
period = int(5e4 / frequency)
data_high = read_raw_data(frequency)
plt.figure(f'{frequency}kHz trace train')
plt.plot(data_high[0, :period * num_traces])


train_100_overlapped = np.zeros(period * (num_traces-1) + 500)
for i in range(num_traces):
    train_100_overlapped[i * period: i * period + 500] += data_100[i, :]

plt.figure(f'Overlapped 100kHz trace train')
plt.plot(train_100_overlapped)