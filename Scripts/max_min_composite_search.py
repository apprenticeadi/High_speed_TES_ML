import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from src.composite_funcs import search_smallest_diff, search_maj_voting, sort_abs_volt_diff
from src.fitting_hist import fitting_histogram

multiplier = 1.2
num_bins = 1000
guess_peak = 30
pca_components = 1  # it's really doubtful if pca helps at all
composite_num = 4

'''
load in data
'''
#calibration data
data_100 = DataUtils.read_raw_data_old(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
offset_cal, _ = calibrationTraces.subtract_offset()

#target data - 'read_high_freq_data' = actual data, uncomment to use
frequency = 600
data_high = calibrationTraces.overlap_to_high_freq(high_frequency=frequency)
'''data_high = DataUtils.read_high_freq_data(frequency)'''
targetTraces = Traces(frequency=frequency, data=data_high, multiplier=multiplier, num_bins=num_bins)
freq_str = targetTraces.freq_str

'''
process calibration data to find range on traces for each photon number using total_traces
'''
total_traces = calibrationTraces.total_traces()
max_photon_number = int((len(total_traces)/3) -1)

'''
apply shift
'''
tar_ave_trace, tar_ave_trace_stdp, tar_ave_trace_stdm = targetTraces.average_trace(plot=False)
shifted_cal_chars = TraceUtils.shift_trace(tar_ave_trace, total_traces, pad_length=guess_peak*2, id=1)

'''
generate composite characteristic traces, using composite_char_traces method
'''
per = len(targetTraces.get_data()[0])
pn_combs, comp_traces = TraceUtils.max_min_trace_utils(shifted_cal_chars, per)
'''
use search methods
'''
target_data = targetTraces.get_data()
pns, errors, tails = search_smallest_diff(target_data, comp_traces, pn_combs)

plt.figure('minimum voltage difference bar')

plt.bar(list(range(len(np.bincount(pns)))), np.bincount(pns))

plt.show()


# <<<<<<<<<<<<<<<<<<< Prepare canvas for plotting errors  >>>>>>>>>>>>>>>>>>
num_rows = (max_photon_number + 1)//2
num_cols = 2
fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 2*num_rows))
fig.canvas.manager.set_window_title('Average abs(voltage difference) from identified characteristic trace')

unique_pns = np.arange(max_photon_number+1)


'''Plot errors'''
ave_errors = errors / targetTraces.period

for i, pn in enumerate(unique_pns):
    row = i // num_cols
    col = i % num_cols

    indices = np.where(pns == pn)
    axs[row, col].hist(ave_errors[indices], bins=100, alpha=0.5, label='Minimum voltage')

    axs[row, col].set_title(f'PN={pn}')
    axs[row, col].set_ylim([0, 300])

# visualise fits
test_num = 8
initial_trace = 0
closest_k = 30  # half the number of composite char traces that will be identified
abs_diff = True # whether we ask for sum(abs(diff)) or sum(diff)

if abs_diff:
    fig_name = f'Identify closest composite trace by sum(abs(diff))'
else:
    fig_name = f'Identify closest composite trace by sum(diff)'

fig = plt.figure(fig_name, figsize=(16, ((test_num+1) // 2)*3))
axgrid = fig.add_gridspec( ((test_num+1) // 2) *2, 8)

for i in range(test_num):
    trace = target_data[initial_trace+i]

    row_num = i // 2
    col_num = i % 2
    ax = fig.add_subplot(axgrid[row_num*2 : (row_num+1)*2, col_num*4 : (col_num+1)*4])

    ax.plot(trace, '--', color='black', label='raw data')

    idx_sort, diffs = sort_abs_volt_diff(trace, comp_traces, k=closest_k)
    # idx_sort, diffs = sort_volt_diff(trace, comp_cal_chars, k=4)  # Here we are able to see why we need absolute value of voltage difference

    for idx in idx_sort:
        plt.plot(comp_traces[idx], label=f'{pn_combs[idx]}')

    ax.legend(loc=1, fontsize='x-small')
    ax.set_ylim([targetTraces.ymin, targetTraces.ymax])
    ax.set_xlim([0, targetTraces.period])
    # ax.set_xlabel('Time (in sample)')
    # ax.set_ylabel('Voltage')
    ax.set_title(f'{initial_trace+i}-th trace')

    if row_num != (test_num-1) // 2:
        ax.set_xticks([])
    if col_num != 0:
        ax.set_yticks([])
print(np.bincount(tails))