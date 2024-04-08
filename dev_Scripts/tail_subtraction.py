import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial
from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from src.tail_funcs import subtract_tails
from scipy.stats import chisquare
# This script is built on Ruidi's original subtraction_analysis script
# %%

multiplier = 0.6
num_bins = 1000
guess_peak = 30
freq_values = np.arange(300,601,100)
chi_vals = []
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 12))
for frequency,ax in zip(freq_values, axs.ravel()):
    print(frequency)
    '''Calibration data'''
    data_100 = DataUtils.read_raw_data_new(100,8)
    calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
    offset_cal, _ = calibrationTraces.subtract_offset()


    '''higher frequency data
    Here I also test tail subtraction method on artificially overlapped higher frequency data. It seems that the method 
    actually makes the stegosaurus worse. 
    '''
    data_high = DataUtils.read_high_freq_data(frequency, 8, new = True)  # unshifted

    targetTraces = Traces(frequency=frequency, data=data_high, multiplier=multiplier, num_bins=num_bins)
    offset_target, _ = targetTraces.subtract_offset()  # shift the data such that the characteristic 0 photon trace has mean 0
    freq_str = targetTraces.freq_str

    '''
    Analysis for 100kHz, to find the characteristic trace for each photon number
    '''

    cal_hist_fit = calibrationTraces.fit_histogram(plot=False)  # fit stegosaurus for calibration data
    cal_chars = calibrationTraces.characteristic_traces_pn(plot=False)  # find characteristic trace for each photon number

    '''
    Analysis for raw higher frequency data
    '''

    tar_hist_fit = targetTraces.fit_histogram(plot=False)
    tar_chars = targetTraces.characteristic_traces_pn(plot=False)
    period = targetTraces.period

    '''
    Scale calibration characteristic traces to the shape of higher frequency data. 
    More specifically, they are scaled to have the peak position of the higher frequency 1-photon 
    characteristic trace. 
    '''
    scaled_cal_chars = TraceUtils.shift_trace(tar_chars[1], cal_chars)

    '''
    Perform tail subtraction
    '''
    shifted_data = targetTraces.get_data()
    subtracted_data, _ = subtract_tails(shifted_data, scaled_cal_chars, guess_peak=guess_peak, plot=False)

    subTraces = Traces(frequency=frequency, data=subtracted_data, multiplier=1.2, num_bins=num_bins)


    binning_index, binning_traces = subTraces.bin_traces(plot=False, fig_name='Histogram of subtracted traces')
    max_pn_steg = max(binning_index.keys())
    num_trace_per_pn = [len(binning_index[pn]) for pn in range(max_pn_steg + 1)]


    def poisson_curve(x, mu, A):
        return A * (mu ** x) * np.exp(-mu) / factorial(np.abs(x))


    ax.bar(list(range(max_pn_steg+1)), num_trace_per_pn)
    fit, cov = curve_fit(poisson_curve,list(range(max_pn_steg+1)) , num_trace_per_pn, p0=[1.5, np.sum(num_trace_per_pn)], maxfev = 2000)
    x = np.linspace(0, max_pn_steg + 1, 100)
    ax.plot(x, poisson_curve(x, fit[0], fit[1]), label='poisson fit', color='r')
    ax.set_title(str(frequency) + 'kHz')
    expected = poisson_curve(list(range(max_pn_steg+1)), fit[0], fit[1])

    chisq = []
    for i in range(len(expected)):
        chi = ((expected[i] - num_trace_per_pn[i])**2) / expected[i]
        chisq.append((chi))
    chi_vals.append(sum(chisq))

plt.show()
plt.plot(freq_values, chi_vals, '+')
np.savetxt('chi_vals_TS.txt', chi_vals)
plt.xlabel('frequency')
plt.ylabel('chi-square')
plt.title('tail subtraction chi-square vals')
plt.show()