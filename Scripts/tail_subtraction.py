import numpy as np
import matplotlib.pyplot as plt

from src.utils import read_high_freq_data, read_raw_data
from src.traces import Traces

# This script is built on Ruidi's original subtraction_analysis script
# %%

multiplier = 0.6
num_bins = 1000

'''Calibration data'''
data_100 = read_raw_data(100)
calibration_traces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
offset_cal, _ = calibration_traces.subtract_offset()
# %%

'''higher frequency data'''
frequency = 600
data_high = read_high_freq_data(frequency)
target_traces = Traces(frequency=frequency, data=data_high, multiplier=multiplier, num_bins=num_bins)
offset_target, _ = target_traces.subtract_offset()  # shift the data such that the characteristic 0 photon trace has mean 0
freq_str = target_traces.freq_str

'''
Analysis for 100kHz, to find the characteristic trace for each photon number
'''

calibration_traces.plot_traces(50)
cal_hist_fit = calibration_traces.fit_histogram(plot=True)  # fit stegosaurus for calibration data
cal_char_traces = calibration_traces.characteristic_traces_pn(plot=True)  # find characteristic trace for each photon number

'''
Analysis for higher frequency data
'''
target_traces.plot_traces(50)
tar_hist_fit = target_traces.fit_histogram(plot=True)
tar_char_traces = target_traces.characteristic_traces_pn(plot=True)

