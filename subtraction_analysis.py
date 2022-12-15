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
data_100_ = np.loadtxt('Data/all_traces_100kHz_middle.txt',
                       delimiter=',', unpack=True)
data_100 = np.transpose(data_100_)
# %%
'''600kHz data'''
data_600_ = np.loadtxt('Data/all_traces_600kHz.txt',
                       delimiter=',', unpack=True)
data_600_ = np.transpose(data_600_)

frequency = 600  # 600kHz data, remember to update this value if you want to use a different frequency


'''
Analysis for 100kHz, to find the characteristic shape of the waveform for each photon number 
'''
# %%
# plot first 100 traces in 100kHZ data
for i in range(100):
    plt.plot(data_100[i][:200])
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.xlim(0, 200)
plt.show()
# %%
# plot average trace
ave_trace = np.mean(data_100, axis=0)

offset = min(ave_trace)
data_100 -= offset

ave_trace = np.mean(data_100, axis=0)
plt.plot(ave_trace)
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.title('average trace')
plt.show()
# %%
overlap_list = [np.dot(ave_trace, trace) for trace in data_100]
heights, bins, _ = plt.hist(overlap_list, bins=1000)

midBins = (bins[1:] + bins[:-1])/2

multiplier = 0.6
hist_fit = fitting_histogram(heights, midBins, overlap_list, multiplier)
lower_list, upper_list = hist_fit.fitting()
numPeaks = hist_fit.numPeaks
binning_index, binning_traces_100 = hist_fit.trace_bin(data_100)

# %%
'''
plotting the first 50 traces for each photon number
'''
for PN in range(numPeaks):
    for trace in binning_traces_100[PN][:50]:
        plt.plot(trace)
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.title('traces for each photon numbers')
plt.show()
# %%
'''
plot the average trace for each photon number
'''
mean_trace_100 = [np.mean(traces, axis=0) for traces in binning_traces_100]
for mean_trace in mean_trace_100:
    plt.plot(mean_trace)
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.title('average trace for each photon numbers')
plt.show()



'''
rest of the analysis is to do with the 600kHz
'''
# %%
'''
splitting the 600kHz data
'''
idealSamples = 5e4/frequency
samples = np.floor(idealSamples)*500
period = int(idealSamples)

period = int( idealSamples )

data_600=[]
for data_set in data_600_:
    for i in range(int(samples/idealSamples)):
        start = int(i*idealSamples)
        if start+period <samples:
            trace = data_set[start:start+period]
            data_600.append( trace )
        else:
            pass

# %%
ave_trace_600 = np.mean(data_600, axis=0)
plt.plot(ave_trace_600)
plt.show()
# %%
for i in range(1000):
    plt.plot(data_600[i])
plt.show()
# %%
'''
fit and bin the 600kHz data
'''
overlap_list_600 = [np.dot(ave_trace_600, amplitude) for amplitude in data_600]
heights_600, bins_600, _ = plt.hist(overlap_list_600, bins=1000)

midBins_600 = (bins_600[1:] + bins_600[:-1])/2
multiplier = 0.6
hist_fit = fitting_histogram(heights_600, midBins_600, overlap_list_600, multiplier)
lower_list, upper_list = hist_fit.fitting()
binning_index_600, binning_traces_600 = hist_fit.trace_bin(data_600)
#%%
'''
shift the data such that the zero photon data have mean overlap=0
'''
mean_0 = np.mean( binning_traces_600[0],axis=0 )
sig_0 = np.sqrt(  np.var(binning_traces_600[0],axis=0)  )
#plt.plot(mean_0)
#plt.plot(sig_0)
offset = min(mean_0)
print(offset)

data_shifted = np.array(data_600) - offset
ave_trace_shifted = ave_trace_600 - offset
#binning_traces_shifted = [ [data_shifted[index] for index in index_list]  for index_list in binning_index]
#%%
overlap_list_shifted = [ np.dot(ave_trace_shifted,amplitude)  for amplitude in data_shifted]
heights_shifted,bins_shifted,_=plt.hist(overlap_list_shifted,bins=1000)

midBins_shifted= (bins_shifted[1:] + bins_shifted[:-1])/2
multiplier = 0.6
hist_fit = fitting_histogram(heights_shifted, midBins_shifted, overlap_list_shifted, multiplier)
lower_list, upper_list = hist_fit.fitting()
binning_index_600, binning_traces_600 = hist_fit.trace_bin(data_shifted)

mean_trace_600 = [np.mean(traces,axis=0)  for traces in binning_traces_600]
for mean in mean_trace_600 :
    plt.plot(mean)
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.title('average trace for each photon numbers')
plt.show()
#%%
'''mapping the mean traces from 100kHz data to 600kHz data'''
# =============================================================================
# max_mean_500 = [max(mean)  for mean in mean_trace_500]
# argmax_mean_500 = [np.argmax(mean)  for mean in mean_trace_500]
# max_mean_100 = [max(mean)  for mean in mean_trace_100]
# mean_trace_100_pad = [ np.insert(mean,0,list(mean[-40:]))  for mean in mean_trace_100]
# argmax_mean_100 = [np.argmax(mean)  for mean in mean_trace_100_pad]
# =============================================================================
mean_trace_100_pad = [ np.insert(mean,0,list(mean[-40:]))  for mean in mean_trace_100]   # pad 40 samples in front of the data to avoid negative index in tail subtraction

diff_max = max(mean_trace_600[1]) / max(mean_trace_100[1])
diff_arg = np.argmax(mean_trace_100_pad[1]) - np.argmax(mean_trace_600[1])

#mean_scaled_100 = [ mean_trace_100_pad[i][(argmax_mean_100[i]-argmax_mean_500[i]) :] /max_mean_100[i] * max_mean_500[i]    for i in range(1,8)]
mean_scaled_100 = [ mean_trace_100_pad[i][diff_arg:]*diff_max     for i in range(1,9)]
mean_scaled_100.insert( 0, mean_trace_100[0] )


plt.plot(mean_scaled_100[0],color='black',label='100kHz trace(after scaling)')
plt.plot(mean_trace_600[0],color='red',label='600kHz trace')
for i in range(1,len(mean_trace_600)):
    plt.plot(mean_scaled_100[i],color='black')
    plt.plot(mean_trace_600[i],color='red')
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.title('average trace for each photon numbers')
plt.legend()
plt.show()
#%%
'''
tail subtraction
'''
subtract=0
mean_max = [max(mean)  for mean in mean_trace_100_pad]
mean_argmax = [np.argmax(mean)  for mean in mean_trace_100_pad]

mean_trace_period = [mean_scaled_100[i][:period] for i in range(len(mean_scaled_100))]
#mean_trace_2period = [mean_scaled_100[i][period:2*period] for i in range(len(mean_scaled_100))]
def mean_trace_subtraction(index,subtract):   
    trace = np.array(data_shifted[index]) - subtract
    diff = [  np.mean( abs(trace - mean_trace_period[i]) )   for i in range(len(mean_scaled_100))]
    PN = np.argmin(diff)  # photon number correspond to this trace
    if PN ==0:
        subtract = 0
        fit=np.zeros(period)
    else:
        trace_max = max(trace[:60])
        trace_argmax = np.argmax(trace[:60])

        offset_diff = mean_argmax[PN] - trace_argmax

        fit = mean_trace_100_pad[PN][ offset_diff :] / mean_max[PN] * trace_max
        subtract = fit[period:2*period]

        
# =============================================================================
#     plt.plot(data_shifted[index])
#     plt.plot(fit[:period],color='red')
#     plt.plot(trace)
#     plt.show()
# =============================================================================
    return trace,subtract

#number_store=[0]
subtracted_trace=[]
for i in range(len(data_shifted)):
    trace,subtract = mean_trace_subtraction(i,subtract)
    subtracted_trace.append(trace)
#%%
ave_subtracted = np.mean(subtracted_trace,axis=0)
plt.plot(ave_subtracted)
plt.ylabel('voltage')
plt.xlabel('time (in sample)')
plt.title('mean trace after tail subtraction')
plt.show()
#%%
overlap = [ np.dot(ave_subtracted,amplitude)  for amplitude in subtracted_trace]
plt.hist(overlap,bins=1000)
plt.ylabel('frequencies')
plt.xlabel('overlap')
plt.title('historgams after subtraction')
plt.show()
