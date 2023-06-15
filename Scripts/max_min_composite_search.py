import numpy as np
import matplotlib.pyplot as plt

from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from src.composite_funcs import search_smallest_diff, search_maj_voting, sort_abs_volt_diff
from src.fitting_hist import fitting_histogram


multiplier = 1.0
num_bins = 1000
guess_peak = 30
pca_components = 1  # it's really doubtful if pca helps at all
composite_num = 3

'''
load in data
'''
#calibration data
data_100 = DataUtils.read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)

#target data - 'read_high_freq_data' = actual data, uncomment to use
frequency = 900
# data_high = calibrationTraces.overlap_to_high_freq(high_frequency=frequency)
data_high = DataUtils.read_high_freq_data(frequency)
targetTraces = Traces(frequency=frequency, data=data_high, multiplier=multiplier, num_bins=num_bins)
freq_str = targetTraces.freq_str
'''
process calibration data to find range on traces for each photon number using total_traces
'''
total_traces = calibrationTraces.total_traces()

'''
apply shift
'''
tar_ave_trace, tar_ave_trace_stdp, tar_ave_trace_stdm = targetTraces.average_trace(plot=False)
shifted_cal_chars = TraceUtils.shift_trace(tar_ave_trace, total_traces, pad_length=guess_peak*2, id=1)

'''
generate composite characteristic traces, using composite_char_traces method
'''
pn_combs, comp_traces = TraceUtils.max_min_trace_utils(shifted_cal_chars)

'''
use search methods
'''
target_data = targetTraces.get_data()
pns, errors = search_smallest_diff(target_data, comp_traces, pn_combs)

plt.figure('minimum voltage difference bar')
plt.bar(list(range(10)), np.bincount(pns))
print(np.bincount(pns))
plt.ylim([0, 6000])
plt.show()


# <<<<<<<<<<<<<<<<<<< Prepare canvas for plotting errors  >>>>>>>>>>>>>>>>>>
num_rows = (9 + 1)//2
num_cols = 2
fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 2*num_rows))
fig.canvas.manager.set_window_title('Average abs(voltage difference) from identified characteristic trace')

unique_pns = np.arange(9+1)


'''Plot errors'''
ave_errors = errors / targetTraces.period

for i, pn in enumerate(unique_pns):
    row = i // num_cols
    col = i % num_cols

    indices = np.where(pns == pn)
    axs[row, col].hist(ave_errors[indices], bins=100, alpha=0.5, label='Minimum voltage')

    axs[row, col].set_title(f'PN={pn}')
    axs[row, col].set_ylim([0, 300])