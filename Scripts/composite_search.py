import numpy as np
import matplotlib.pyplot as plt

from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from src.composite_funcs import identify_by_area_diff

multiplier = 0.6
num_bins = 1000
guess_peak = 30
pca_components = 1  # it's really doubtful if pca helps at all
composite_num = 3


# <<<<<<<<<<<<<<<<<<< Calibation data  >>>>>>>>>>>>>>>>>>
data_100 = DataUtils.read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)

'''Shift data such that 0-photon trace has mean 0'''
offset_cal, _ = calibrationTraces.subtract_offset()

'''PCA cleanup calibration data'''
# calibrationTraces.pca_cleanup(num_components=pca_components)

'''Find characteristic trace for each photon number'''
cal_chars = calibrationTraces.characteristic_traces_pn(plot=False)  # find characteristic trace for each photon number


# <<<<<<<<<<<<<<<<<<< Target data  >>>>>>>>>>>>>>>>>>
frequency = 600
data_high = DataUtils.read_high_freq_data(frequency)  # unshifted
targetTraces = Traces(frequency=frequency, data=data_high, multiplier=multiplier, num_bins=num_bins)
freq_str = targetTraces.freq_str

'''Shift data'''
offset_target, _ = targetTraces.subtract_offset()

'''PCA cleanup'''
# _ = targetTraces.pca_cleanup(num_components=pca_components)

# <<<<<<<<<<<<<<<<<<< Calibration characteristic traces  >>>>>>>>>>>>>>>>>>
'''Shift calibration characteristic traces'''
tar_ave_trace = targetTraces.average_trace(plot=False)
shifted_cal_chars = TraceUtils.shift_trace(tar_ave_trace, cal_chars, pad_length=guess_peak*2, id=1)

plt.figure('Shifted char traces')
plt.plot(tar_ave_trace, color='red', label=f'{freq_str} overall average trace')
for i in range(len(shifted_cal_chars)):
    if i==0:
        plt.plot(shifted_cal_chars[i], color='black', label='100kHz shifted char traces')
    else:
        plt.plot(shifted_cal_chars[i], color='black')
plt.xlim([0, composite_num * targetTraces.period])
plt.ylim([targetTraces.ymin, targetTraces.ymax])
plt.legend()

'''Find the composite characteristic traces'''
pn_tuples, comp_cal_chars = TraceUtils.composite_char_traces(shifted_cal_chars, targetTraces.period, comp_num=composite_num)

plt.figure(f'{composite_num}-composite char traces')
for i, pn_tuple in enumerate(pn_tuples):
    if np.max(pn_tuple) <= 4:
        plt.plot(comp_cal_chars[i], label=f'{pn_tuple}')

# <<<<<<<<<<<<<<<<<<< Perform composite search  >>>>>>>>>>>>>>>>>>
target_data = targetTraces.get_data()

'''Test and plot the method'''
test_num = 8
initial_trace = 3000
closest_k = 4  # half the number of composite char traces that will be identified
fig = plt.figure("Identify trace number by composite characteristic traces", figsize=(16, ((test_num+1) // 2)*3))
axgrid = fig.add_gridspec( ((test_num+1) // 2) *2, 8)
for i in range(test_num):
    trace = target_data[initial_trace+i]

    row_num = i // 2
    col_num = i % 2
    ax = fig.add_subplot(axgrid[row_num*2 : (row_num+1)*2, col_num*4 : (col_num+1)*4])

    ax.plot(trace, '--', color='black', label='raw data')

    idx_sort, diffs = identify_by_area_diff(trace, comp_cal_chars, abs=False, k=closest_k)

    for idx in idx_sort:
        plt.plot(comp_cal_chars[idx], label=f'{pn_tuples[idx]}')

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


# TODO: How to identify photon number, and how to benchmark uncertainty? Just like in stegosaurus?
# Here's a thought: Identify the photon number of each trace, and keep record of the diff=mean(abs(difference)). In the
# end, plot a histogram of this diff. The closer this histogram is concentrated to 0 the better. And compare this with
# dot product stegosaurus method.




