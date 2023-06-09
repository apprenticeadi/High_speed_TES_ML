import numpy as np
import matplotlib.pyplot as plt

from src.utils import read_high_freq_data, read_raw_data
from src.tail_funcs import shift_trace, pad_trace, composite_char_traces
from src.traces import Traces

multiplier = 0.6
num_bins = 1000
guess_peak = 30
pca_components = 2
composite_num = 3


# <<<<<<<<<<<<<<<<<<< Calibation data  >>>>>>>>>>>>>>>>>>
data_100 = read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)

'''Shift data such that 0-photon trace has mean 0'''
offset_cal, _ = calibrationTraces.subtract_offset()

'''Find characteristic trace for each photon number'''
cal_chars = calibrationTraces.characteristic_traces_pn(plot=False)  # find characteristic trace for each photon number


# <<<<<<<<<<<<<<<<<<< Target data  >>>>>>>>>>>>>>>>>>
frequency = 600
data_high = read_high_freq_data(frequency)  # unshifted
targetTraces = Traces(frequency=frequency, data=data_high, multiplier=multiplier, num_bins=num_bins)
freq_str = targetTraces.freq_str

'''Shift data'''
offset_target, _ = targetTraces.subtract_offset()

'''PCA cleanup'''
_ = targetTraces.pca_cleanup(num_components=pca_components)

# <<<<<<<<<<<<<<<<<<< Calibration characteristic traces  >>>>>>>>>>>>>>>>>>
'''Shift calibration characteristic traces'''
tar_ave_trace = targetTraces.average_trace(plot=False)
shifted_cal_chars = shift_trace(tar_ave_trace, cal_chars, pad_length=guess_peak*2, id=1)

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
pn_pairs, comp_cal_chars = composite_char_traces(shifted_cal_chars, targetTraces.period, comp_num=composite_num)

plt.figure(f'{composite_num}-composite char traces')
for i, pn_pair in enumerate(pn_pairs):
    if np.max(pn_pair) <= 4:
        plt.plot(comp_cal_chars[i], label=f'{pn_pair}')

# <<<<<<<<<<<<<<<<<<< Perform composite search  >>>>>>>>>>>>>>>>>>
target_data = targetTraces.get_data()

'''Test and plot the method'''
test_num = 10
initial_trace = 1000
closest_k = 5
fig = plt.figure("Identify trace number by composite characteristic traces", figsize=(20, 16))
axgrid = fig.add_gridspec(10, 8)
for i in range(test_num):
    trace = target_data[initial_trace+i]

    row_num = i // 2
    col_num = i % 2
    ax = fig.add_subplot(axgrid[row_num*2 : (row_num+1)*2, col_num*4 : (col_num+1)*4])

    ax.plot(trace, label='raw data')

    diff = np.mean(np.abs(trace - comp_cal_chars), axis=1)
    idx_sort = np.argpartition(diff, closest_k)


    for idx in idx_sort[:closest_k]:
        plt.plot(comp_cal_chars[idx], label=f'{pn_pairs[idx]}')

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




