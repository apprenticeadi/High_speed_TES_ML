import numpy as np
import matplotlib.pyplot as plt

from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from src.method_funcs import subtract_tails

# This script is built on Ruidi's original subtraction_analysis script
# %%

multiplier = 0.6
num_bins = 1000
guess_peak = 30

'''Calibration data'''
data_100 = DataUtils.read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
offset_cal, _ = calibrationTraces.subtract_offset()
# %%

'''higher frequency data
Here I also test tail subtraction method on artificially overlapped higher frequency data. It seems that the method 
actually makes the stegosaurus worse. 
'''
frequency = 700
data_high = DataUtils.read_high_freq_data(frequency)  # unshifted

# data_high = calibrationTraces.overlap_to_high_freq(high_frequency=frequency)
targetTraces = Traces(frequency=frequency, data=data_high, multiplier=multiplier, num_bins=num_bins)
offset_target, _ = targetTraces.subtract_offset()  # shift the data such that the characteristic 0 photon trace has mean 0
freq_str = targetTraces.freq_str

'''
Analysis for 100kHz, to find the characteristic trace for each photon number
'''

# calibrationTraces.plot_traces(50)
cal_hist_fit = calibrationTraces.fit_histogram(plot=True)  # fit stegosaurus for calibration data
cal_chars = calibrationTraces.characteristic_traces_pn(plot=True)  # find characteristic trace for each photon number

'''
Analysis for raw higher frequency data
'''
targetTraces.plot_traces(10)
targetTraces.plot_trace_trains(num_trains=1, num_traces=10)
# The hope is the fit on raw data is accurate for the 0 and 1 photon.
tar_hist_fit = targetTraces.fit_histogram(plot=True)
tar_chars = targetTraces.characteristic_traces_pn(plot=False)
period = targetTraces.period

'''
Scale calibration characteristic traces to the shape of higher frequency data. 
More specifically, they are scaled to have the peak position of the higher frequency 1-photon 
characteristic trace. 
'''
scaled_cal_chars = TraceUtils.shift_trace(tar_chars[1], cal_chars)

plt.figure('Scaled calibration characteristic traces')
for i in range(len(scaled_cal_chars)):
    if i == 0:
        plt.plot(scaled_cal_chars[i], color='black', label='100kHz')
    else:
        plt.plot(scaled_cal_chars[i], color='black')
for i in range(len(tar_chars)):
    if i==0:
        plt.plot(tar_chars[i], color='red', label=f'{frequency}kHz')
    else:
        plt.plot(tar_chars[i],color='red')
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.xlim([0, 100])
plt.ylim([-3000, 35000])
plt.title('average trace for each photon numbers')
plt.legend()

'''
Perform tail subtraction
'''
shifted_data = targetTraces.get_data()
subtracted_data, _ = subtract_tails(shifted_data, scaled_cal_chars, guess_peak=guess_peak, plot=True)

subTraces = Traces(frequency=frequency, data=subtracted_data, multiplier=multiplier, num_bins=num_bins)
# subTraces.plot_traces(50, fig_name='First 50 of subtracted traces')
sub_histfit = subTraces.fit_histogram(plot=True, fig_name='Histogram of subtracted traces')