#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:25:35 2022

@author: ruidizhu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from fitting_hist import fitting_histogram

# %%
data_100_ = np.loadtxt(r'../Data/all_traces_100kHz_middle.txt',
                       delimiter=',', unpack=True)
data_100 = np.transpose(data_100_)


'''
Analysis for 100kHz, to find the characteristic shape of the waveform for each photon number 
'''
# %%
min_voltage = np.amin(data_100)
max_voltage = np.amax(data_100)
ymin = 5000 * (min_voltage // 5000)
ymax = 5000 * (max_voltage // 5000 + 1)

'''
plot first x traces in 100kHZ data
'''
num_traces = 1000
plt.figure('100kHz traces')
for i in range(num_traces):
    plt.plot(data_100[i][:200])
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.xlim(0, 200)
plt.ylim(ymin, ymax)

# # %%
# '''
# plot average trace
# '''
#
# ave_trace = np.mean(data_100, axis=0)
# #
# offset = min(ave_trace)
# data_100 -= offset
#
# plt.figure('average trace 100kHz')
# ave_trace = np.mean(data_100, axis=0)
# plt.plot(ave_trace)
# plt.ylabel('voltage')
# plt.xlabel('time (in sample)')
# plt.ylim(ymin, ymax)
# plt.title('average trace')
#
# # # %%
# '''
# 100kHz stegosaurus
# '''
# plt.figure('100 kHz stegosaurus')
# overlaps = ave_trace @ data_100.T
# heights, bins, _ = plt.hist(overlaps, bins=1000, color='aquamarine')
#
# midBins = (bins[1:] + bins[:-1]) / 2
#
# multiplier = 0.6
# hist_fit = fitting_histogram(heights, midBins, overlaps, multiplier)
# lower_list, upper_list = hist_fit.fitting(figname='fit 100kHz')
# numPeaks = hist_fit.numPeaks
# binning_index, binning_traces_100 = hist_fit.trace_bin(data_100)
#
# # # %%
# '''
# plotting the first x traces for each photon number
# '''
#
# for PN in range(numPeaks):
#     plt.figure(f'{PN} photon traces')
#     for trace in binning_traces_100[PN][:50]:
#         plt.plot(trace)
#     plt.ylim(ymin, ymax)
#     plt.ylabel('voltage')
#     plt.xlabel('time (in sample)')
#     plt.title(f'traces for {PN}')
# plt.show()
# # %%
# '''
# plot the average trace for each photon number
# '''
# mean_trace_100 = {}
# for PN in range(numPeaks):
#     mean_trace = np.mean(binning_traces_100[PN], axis=0)
#     mean_trace_100[PN] = mean_trace
#     # plt.figure(f'{PN} photon traces')
#     # plt.plot(mean_trace, 'k--', label='mean trace')
#     # plt.legend()
#
#     plt.figure('Mean traces')
#     plt.plot(mean_trace, label=f'{PN} photons')
#     plt.ylabel('voltage')
#     plt.xlabel('time (in sample)')
#     plt.title('average trace for each photon numbers')
