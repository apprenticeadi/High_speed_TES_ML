import numpy as np
import matplotlib.pyplot as plt

from src.utils import read_high_freq_data, read_raw_data
from src.tail_funcs import shift_trace, pad_trace, composite_char_traces
from src.traces import Traces

multiplier = 0.6
num_bins = 1000
guess_peak = 30

'''Calibration data'''
data_100 = read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
offset_cal, _ = calibrationTraces.subtract_offset()

cal_chars = calibrationTraces.characteristic_traces_pn(plot=True)  # find characteristic trace for each photon number

# %%

'''higher frequency data'''
frequency = 600
data_high = read_high_freq_data(frequency)  # unshifted

targetTraces = Traces(frequency=frequency, data=data_high, multiplier=multiplier, num_bins=num_bins)
offset_target, data_high_shifted = targetTraces.subtract_offset()  # shift the data such that the characteristic 0 photon trace has mean 0
freq_str = targetTraces.freq_str
tar_ave_trace = targetTraces.average_trace(plot=True)

'''shift calibration characteristic traces to have the same peak position as the target average trace'''
shifted_cal_chars = shift_trace(tar_ave_trace, cal_chars, id=1)

plt.figure('Shifted char traces')
plt.plot(tar_ave_trace, color='red', label=f'{freq_str} overall average trace')
for i in range(len(shifted_cal_chars)):
    if i==0:
        plt.plot(shifted_cal_chars[i], color='black', label='100kHz shifted char traces')
    else:
        plt.plot(shifted_cal_chars[i], color='black')
plt.xlim([0, targetTraces.period])
plt.ylim([targetTraces.ymin, targetTraces.ymax])
plt.legend()

'''find the composite characteristic traces'''
pn_pairs, comp_cal_chars = composite_char_traces(shifted_cal_chars, targetTraces.period)

plt.figure('Composite char traces')
for i, pn_pair in enumerate(pn_pairs):
    if np.max(pn_pair) <=4:
        plt.plot(comp_cal_chars[i], label=f'{pn_pair}')

'''identify target trace photon number'''
test_num = 10
closest_k = 5
for i in range(test_num):
    trace = data_high_shifted[i]
    plt.figure(f'{i}-th trace')
    plt.plot(trace, label='raw data')

    diff = np.mean(np.abs(trace - comp_cal_chars), axis=1)
    idx_sort = np.argpartition(diff, closest_k)


    for idx in idx_sort[:closest_k]:
        plt.plot(comp_cal_chars[idx], label=f'{pn_pairs[idx]}')

    plt.legend()
    plt.ylim([targetTraces.ymin, targetTraces.ymax])
    plt.xlim([0, targetTraces.period])

# TODO: How to identify photon number, and how to benchmark uncertainty? Just like in stegosaurus?
