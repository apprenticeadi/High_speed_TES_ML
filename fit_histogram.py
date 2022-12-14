#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:32:50 2022

@author: ruidizhu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
#%%
def finding_peaks(midBins,heights,plot=True):
    '''
    use data smoothing method to find the peak positions of each bins of the histogram, 
    will be useful for fitting the historgam
    '''
    kernel_size = 10
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved_10 = np.convolve(heights, kernel, mode='same')
    
    plt.plot(midBins,data_convolved_10)
    peaks, _ = find_peaks(data_convolved_10, height=2,distance=30,prominence=1, width=10)
    amplitudes = data_convolved_10[peaks]
    
    #heights,bins,_ = plt.hist(overlap_list,bins=1000,range=(-0.1e10,1.5e10))
    plt.plot(midBins[peaks], amplitudes, "x")
    plt.plot(np.zeros_like(data_convolved_10), "--", color="gray")
    plt.show()
    return peaks, amplitudes

#%%
data_100_ = np.loadtxt('Data/all_traces_100kHz_middle.txt',delimiter=',',unpack=True)
data_100=np.transpose(data_100_)

#%%
for i in range(100):
    plt.plot(data_100[i][:200])
    
#%%
ave_trace = np.mean(data_100,axis=0)

offset = min(ave_trace)
data_100 -=offset

ave_trace = np.mean(data_100,axis=0)
plt.plot(ave_trace)
#%%
overlap_list = [np.dot(ave_trace,trace)  for trace in data_100]
heights,bins,_ = plt.hist(overlap_list,bins=1000,range=(-0.1e10,1.5e10))

midBins = ( bins[1:] + bins[:-1] )/2
# =============================================================================
# diffPositions = ( midBins[1:] + midBins[:-1] )/2
# diff = np.diff(heights)
# plt.plot(diffPositions,diff)
# =============================================================================

peaks, amplitudes = finding_peaks(midBins,heights,plot=True)

positions = midBins[peaks]
sigma = ( positions[1] - positions[0] ) /4
numPeaks = len(peaks)

#%%
def func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params)-1, 3):
        mu = params[i]
        amp = params[i+1]
        sig = params[i+2]
        y = y + amp * np.exp( -((x - mu)/abs(sig))**2)
    return y

guess = []
for i in range(numPeaks):
    guess += [ positions[i] , amplitudes[i] , sigma]
    
mid_positions = ( bins[:-1] +  bins[1:] )/2
popt, pcov = curve_fit(func, mid_positions, heights, p0=guess)

sort = np.argsort(popt[::3])

sorted_popt=[]
for i in sort:
    count=3*i
    popt[count+2] = abs(popt[count+2])
    print(f'{i}th peak:',popt[count:count+3])
    sorted_popt.append(list(popt[count:count+3]) )
sorted_popt = list(np.array(sorted_popt).flat)

x=np.linspace(min(bins),max(bins),10000)
fit = func(x, *sorted_popt)
plt.figure(figsize=(8,5))
heights,bins,_ = plt.hist(overlap_list,bins=1000,range=(-0.1e10,1.5e10))
plt.plot(x,fit)
plt.xlabel('overlap',size=14)
plt.ylabel('entries',size=14)

multiple=0.6
upper_list=[]
lower_list=[]
for i in range(0,numPeaks):
    mu = sorted_popt[i*3]
    width = sorted_popt[i*3+2] * multiple
    upper = mu + width
    lower = mu - width
    plt.vlines(upper,0,func(upper,*sorted_popt),color='red')
    plt.vlines(lower,0,func(lower,*sorted_popt),color='red')
    upper_list.append(upper)
    lower_list.append(lower)
plt.show()



#%%
'''500kHz data'''
data_500_ = np.loadtxt('Data/all_traces_600kHz.txt',delimiter=',',unpack=True)
data_500_=np.transpose(data_500_)
#data_500 = data_500_.reshape(50*int(len(data_500_[0])/83),83)
#%%
samples = 83*500
idealSamples = 500/6
period = int( idealSamples )

data_500=[]
for data_set in data_500_:
    for i in range(int(samples/idealSamples)):
        start = int(i*idealSamples)
        if start+period <samples:
            trace = data_set[start:start+period]
            data_500.append( trace )
        else:
            pass

#%%
ave_trace = np.mean(data_500,axis=0)

overlap_list = [ np.dot(ave_trace,amplitude)  for amplitude in data_500]
heights,bins,_=plt.hist(overlap_list,bins=1000)
#%%
midBins = ( bins[1:] + bins[:-1] )/2

kernel_size = 10
kernel = np.ones(kernel_size) / kernel_size
data_convolved_10 = np.convolve(heights, kernel, mode='same')

plt.plot(midBins,data_convolved_10)
peaks, _ = find_peaks(data_convolved_10, height=2,distance=50,prominence=1, width=10)

heights,bins,_ = plt.hist(overlap_list,bins=1000)
plt.plot(midBins[peaks], data_convolved_10[peaks], "x")
plt.plot(np.zeros_like(data_convolved_10), "--", color="gray")
plt.show()

amplitudes = data_convolved_10[peaks]
positions = midBins[peaks]
sigma = ( positions[1] - positions[0] ) /4

noPeaks = len(peaks)
print(noPeaks)
#%%
def func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params)-1, 3):
        mu = params[i]
        amp = params[i+1]
        sig = params[i+2]
        y = y + amp * np.exp( -((x - mu)/sig)**2)
    return y

guess = []
#amp=[40,110,150,145,110,70,35,15,5]
#po = [0e8,6.36e9,1.23e10,1.78e10,2.28e10,2.76e10,3.13e10,3.3e10]
for i in range(noPeaks):
    #guess += [0 + 0.14e10*i, amp[i], 0.035e10]
    #guess += [po[i], amp[i], 1.1e9]
    guess += [ positions[i] , amplitudes[i] , sigma]
    
mid_positions = ( bins[:-1] +  bins[1:] )/2
popt, pcov = curve_fit(func, mid_positions, heights, p0=guess, bounds=[-0.2e10,1e10])
for i in range(0,noPeaks):
    count=3*i
    print(f'{i}th peak:',popt[count:count+3])
    
x=np.linspace(min(bins),max(bins),10000)
fit = func(x, *popt)
plt.figure(figsize=(8,5))
plt.hist(overlap_list,bins=1000)
plt.plot(x,fit)
plt.xlabel('overlap',size=14)
plt.ylabel('entries',size=14)

multiple=0.5
upper_list=[]
lower_list=[]
for i in range(0,noPeaks):
    mu = popt[i*3]
    width = popt[i*3+2] * multiple
    upper = mu + width
    lower = mu - width
    plt.vlines(upper,0,func(upper,*popt),color='red')
    plt.vlines(lower,0,func(lower,*popt),color='red')
    upper_list.append(upper)
    lower_list.append(lower)
plt.show()
    
