#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:25:35 2022

@author: ruidizhu
"""
import numpy as np
import matplotlib.pyplot as plt

from src.fitting_hist import fitting_histogram
from src.utils import read_raw_data, read_high_freq_data

# %%
# data_100_ = np.loadtxt('Data/all_traces_100kHz_middle.txt',
#                        delimiter=',', unpack=True)
# data_100 = np.transpose(data_100_)

data_100 = read_raw_data(100)
# %%
'''600kHz data'''
# data_600_ = np.loadtxt('Data/all_traces_600kHz.txt',
#                        delimiter=',', unpack=True)
# data_600_ = np.transpose(data_600_)

frequency = 600  # 600kHz data, remember to update this value if you want to use a different frequency
data_600 = read_high_freq_data(frequency)

idealSamples = 5e4 / frequency
samples = np.floor(idealSamples) * 500  # This should be the length of row of data_600_
period = int(idealSamples)

'''
Analysis for 100kHz, to find the characteristic shape of the waveform for each photon number
'''
# %%
min_voltage = np.amin(data_100)
max_voltage = np.amax(data_100)
ymin = 5000 * (min_voltage // 5000)
ymax = 5000 * (max_voltage // 5000 + 1)

'''
plot first _ traces in 100kHZ data
'''
plt.figure('100kHz traces')
for i in range(100):
    plt.plot(data_100[i][:200])
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.xlim(0, 200)
plt.ylim(ymin, ymax)
# plt.show()
# %%

'''
plot average trace
'''
ave_trace = np.mean(data_100, axis=0)
#
offset = min(ave_trace)
data_100 -= offset

plt.figure('average trace 100kHz')
ave_trace = np.mean(data_100, axis=0)
plt.plot(ave_trace)
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.ylim(ymin, ymax)
plt.title('average trace')
# plt.show()
# # %%

'''
100kHz stegosaurus
'''
plt.figure('100 kHz stegosaurus')
overlaps = ave_trace @ data_100.T
heights, bins, _ = plt.hist(overlaps, bins=1000, color='aquamarine')

midBins = (bins[1:] + bins[:-1]) / 2

multiplier = 0.6
hist_fit = fitting_histogram(heights, midBins, overlaps, multiplier)
lower_list, upper_list = hist_fit.fitting(figname='fit 100kHz')
numPeaks = hist_fit.numPeaks

binning_index, binning_traces_100 = hist_fit.trace_bin(data_100)

# # %%
'''
plotting the first x traces for each photon number
'''
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
'''
plot the average trace for each photon number
'''
mean_trace_100 = {}

for PN in range(numPeaks):
    mean_trace = np.mean(binning_traces_100[PN], axis=0)
    mean_trace_100[PN] = mean_trace
    # plt.figure(f'{PN} photon traces')
    # plt.plot(mean_trace, 'k--', label='mean trace')
    # plt.legend()

    plt.figure('Mean traces')
    plt.plot(mean_trace, label=f'{PN} photons')
    plt.ylabel('voltage')
    plt.xlabel('time (in sample)')
    plt.title('average trace for each photon numbers')
    plt.legend()
# # plt.show()
'''
rest of the analysis is to do with the 600kHz
'''
# # # %%
# '''
# splitting the 600kHz data
# 100 kHz data corresponds to 10us period, which is represented by 500 datapoints per trace. The time between two
# datapoints is thus 10ns.
# For 600kHz, the number of datapoints per trace should be 500 * (100kHz / 600kHz) = 83.33.
# However, when Ruidi tried to save 83 datapoints per trace, for some reason the traces were not continuous, which makes
# tail subtraction impossible. So instead Ruidi saved 500*83 datapoints per row.
#
# This section manually splits into the correct trace lengths.
# '''
#
# idealSamples = 5e4 / frequency
# samples = np.floor(idealSamples) * 500  # This should be the length of row of data_600_
# period = int(idealSamples)
#
# data_600 = []
# for data_set in data_600_:
#     for i in range(1, 499):  # skip the first trace and last trace
#
#         start = int(i * idealSamples)
#         if start + period < samples:
#             trace = data_set[start:start + period]
#             data_600.append(trace)
#         else:
#             pass
# data_600 = np.asarray(data_600)

# # %%
plt.figure(f'average {frequency}kHz trace')
ave_trace_600 = np.mean(data_600, axis=0)
plt.plot(ave_trace_600)
# plt.show()
# # %%
'''
Plot first x traces of 600kHz
'''
num_600_traces = 100
plt.figure(f'First {num_600_traces} traces of {frequency}kHz')
for i in range(num_600_traces):
    plt.plot(data_600[i])
plt.xlim(0, period)
# plt.show()
# # %%
'''
fit and bin the 600kHz data, here only the 0 photon trace matter
'''
plt.figure(f'{frequency}kHz raw stegosaurus')
overlaps_600 = ave_trace_600 @ data_600.T  # [np.dot(ave_trace_600, amplitude) for amplitude in data_600]
heights_600, bins_600, _ = plt.hist(overlaps_600, bins=1000)

midBins_600 = (bins_600[1:] + bins_600[:-1]) / 2
multiplier = 0.6
hist_fit = fitting_histogram(heights_600, midBins_600, overlaps_600, multiplier)
lower_list, upper_list = hist_fit.fitting(figname=f'fit {frequency}kHz raw')
binning_index_600, binning_traces_600 = hist_fit.trace_bin(data_600)
# %%
'''
shift the data such that the zero photon data have mean overlap=0
'''
mean_0 = np.mean(binning_traces_600[0], axis=0)  # voltage mean value of zero photon traces
sig_0 = np.sqrt(np.var(binning_traces_600[0], axis=0))
plt.figure(f'{frequency}kHz 0-photon traces mean and std')
plt.plot(mean_0, label='mean')
plt.plot(sig_0, label='std')
plt.legend()
offset = min(mean_0)
print(offset)

data_shifted = np.array(data_600) - offset
ave_trace_shifted = ave_trace_600 - offset

# binning_traces_shifted = {}
# for photon_number in binning_traces_600.keys():
#     binning_traces_shifted[photon_number] = binning_traces_600[photon_number] - offset

# %%p
overlap_list_shifted = ave_trace_shifted @ data_shifted.T
# overlap_list_shifted_2 = [ np.dot(ave_trace_shifted,amplitude)  for amplitude in data_shifted]  # calculate overlap between shifted traces and shifted mean trace
plt.figure(f'{frequency}kHz shifted stegosaurus')
heights_shifted, bins_shifted, _ = plt.hist(overlap_list_shifted, bins=1000)

'''
Fit the shifted data but only the 1-photon peak is important here. 
'''
midBins_shifted = (bins_shifted[1:] + bins_shifted[:-1]) / 2
multiplier = 0.6
hist_fit = fitting_histogram(heights_shifted, midBins_shifted, overlap_list_shifted, multiplier)
lower_list, upper_list = hist_fit.fitting(figname=f'fit {frequency}kHz shifted')
binning_index_600, binning_traces_600 = hist_fit.trace_bin(data_shifted)

# The mean photon traces for higher photon numbers are likely to be incorrect.
plt.figure(f'{frequency}kHz average shifted trace for each photon number')
mean_trace_600 = {}
for photon_number in binning_traces_600.keys():
    mean = np.mean(binning_traces_600[photon_number], axis=0)
    mean_trace_600[photon_number] = mean
    plt.plot(mean, label=f'{photon_number} photons')
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.title('average trace for each photon numbers')
plt.xlim(0, 500)
plt.legend()
# plt.show()
# %%
'''mapping the mean traces from 100kHz data to 600kHz data'''
# =============================================================================
# max_mean_500 = [max(mean)  for mean in mean_trace_500]
# argmax_mean_500 = [np.argmax(mean)  for mean in mean_trace_500]
# max_mean_100 = [max(mean)  for mean in mean_trace_100]
# mean_trace_100_pad = [ np.insert(mean,0,list(mean[-40:]))  for mean in mean_trace_100]
# argmax_mean_100 = [np.argmax(mean)  for mean in mean_trace_100_pad]
# =============================================================================

mean_trace_100_pad = {}
for photon_number in mean_trace_100.keys():
    mean_trace_100_pad[photon_number] = np.insert(mean_trace_100[photon_number], 0, mean_trace_100[photon_number][-40:])

plt.figure('100kHz mean trace pad')
for i in mean_trace_100_pad.keys():
    plt.plot(mean_trace_100_pad[i], label=f'{i}')
plt.ylabel('voltage')
plt.xlabel('time (padded)')
plt.title('100kHz mean trace pads')


diff_max = max(mean_trace_600[1]) / max(mean_trace_100[1])
diff_arg = np.argmax(mean_trace_100_pad[1]) - np.argmax(mean_trace_600[1])  # the position of the peak to map the time range of 100khz to that of 600khz

#mean_scaled_100 = [ mean_trace_100_pad[i][(argmax_mean_100[i]-argmax_mean_500[i]) :] /max_mean_100[i] * max_mean_500[i]    for i in range(1,8)]
mean_scaled_100 = {}
for photon_number in mean_trace_100_pad.keys():
    if photon_number == 0:
        mean_scaled_100[photon_number] = mean_trace_100[photon_number]

    else:
        mean_scaled_100[photon_number] = mean_trace_100_pad[photon_number][diff_arg:] * diff_max


plt.figure('Mean trace for two frequencies')
for i in mean_scaled_100.keys():
    if i == 0:
        plt.plot(mean_scaled_100[i], color='black', label='100kHz')
    else:
        plt.plot(mean_scaled_100[i], color='black')
for i in mean_trace_600.keys():
    if i==0:
        plt.plot(mean_trace_600[i], color='red', label=f'{frequency}kHz')
    else:
        plt.plot(mean_trace_600[i],color='red')
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.xlim([0, period])
plt.ylim([0, 35000])
plt.title('average trace for each photon numbers')
plt.legend()
# plt.show()
# #%%
'''
tail subtraction
'''
subtract=0
mean_max = [max(mean)  for mean in mean_trace_100_pad.values()]
mean_argmax = [np.argmax(mean)  for mean in mean_trace_100_pad.values()]

mean_trace_period = [mean_scaled_100[i][:period] for i in mean_scaled_100.keys()]
#mean_trace_2period = [mean_scaled_100[i][period:2*period] for i in range(len(mean_scaled_100))]
def mean_trace_subtraction(index,subtract):
    trace = np.array(data_shifted[index]) - subtract  # The subtract is updated everytime by previous trace
    diff = [  np.mean( abs(trace - mean_trace_period[i]) )   for i in range(len(mean_scaled_100))]

    #TODO: it's not clear to me that this is a valid way of identifying the photon number of the trace
    PN = np.argmin(diff)  # photon number correspond to this trace
    if PN ==0:
        subtract = 0
        fit=np.zeros(period)
    else:
        trace_max = max(trace[:60])
        trace_argmax = np.argmax(trace[:60])

        offset_diff = mean_argmax[PN] - trace_argmax

        fit = mean_trace_100_pad[PN][ offset_diff :] / mean_max[PN] * trace_max   # fit the tail of the corresponding 100kHz mean trace , but rescaled to have the same peak position and height
        subtract = fit[period:2*period]


# =============================================================================
    if index >=5 and index <10:
        plt.figure(f'{index} trace subtraction')
        plt.plot(data_shifted[index], label='shifted raw data')
        plt.plot(fit[:period], label='fit')
        plt.plot(trace, label='subtracted trace')
        plt.plot(subtract, label='tail to be subtracted from the next one')
        plt.plot(mean_trace_period[PN], label='identified photon number mean trace')
        plt.ylim(top = 35000)
        plt.legend()
        plt.show()
# =============================================================================
    return trace,subtract

#number_store=[0]
subtracted_trace=[]
for i in range(len(data_shifted)):
    trace,subtract = mean_trace_subtraction(i,subtract)
    subtracted_trace.append(trace)
#%%
ave_subtracted = np.mean(subtracted_trace,axis=0)
plt.figure('Mean trace after tail subtraction')
plt.plot(ave_subtracted)
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.title('mean trace after tail subtraction')
# plt.show()
#%%
plt.figure('histogram after subtraction')
overlap = [ np.dot(ave_subtracted,amplitude)  for amplitude in subtracted_trace]
plt.hist(overlap,bins=1000)
plt.ylabel('frequencies')
plt.xlabel('overlap')
plt.title('historgams after subtraction')
# plt.show()
