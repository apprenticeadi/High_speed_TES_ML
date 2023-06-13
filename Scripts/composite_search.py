import numpy as np
import matplotlib.pyplot as plt
import time


from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from src.composite_funcs import sort_volt_diff, sort_abs_volt_diff, search_smallest_diff, search_maj_voting
from src.tail_funcs import subtract_tails_batch

multiplier = 1.2
num_bins = 1000
guess_peak = 30
pca_components = 1  # it's really doubtful if pca helps at all
composite_num = 4


# <<<<<<<<<<<<<<<<<<< Calibation data  >>>>>>>>>>>>>>>>>>
data_100 = DataUtils.read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)

'''Shift data such that 0-photon trace has mean 0'''
offset_cal, _ = calibrationTraces.subtract_offset()

'''PCA cleanup calibration data'''
# calibrationTraces.pca_cleanup(num_components=pca_components)

'''Histogram'''
calibrationTraces.fit_histogram(plot=False)
calibrationTraces.pn_bar_plot()

'''Find characteristic trace for each photon number'''
cal_chars = calibrationTraces.characteristic_traces_pn(plot=False)
# find characteristic trace for each photon number
max_photon_number = len(cal_chars) - 1

# <<<<<<<<<<<<<<<<<<< Target data  >>>>>>>>>>>>>>>>>>
frequency = 900
#data_high = calibrationTraces.overlap_to_high_freq(high_frequency=frequency)
data_high = DataUtils.read_high_freq_data(frequency)  # unshifted
targetTraces = Traces(frequency=frequency, data=data_high, multiplier=multiplier, num_bins=num_bins)
freq_str = targetTraces.freq_str

'''Shift data'''
# I'm actually not sure if I should do this or not... It might give the traces an incorrect height.
offset_target, _ = targetTraces.subtract_offset()

'''PCA cleanup'''
# _ = targetTraces.pca_cleanup(num_components=pca_components)

'''Raw histogram by inner product'''
targetTraces.fit_histogram(plot=False)
targetTraces.characteristic_traces_pn(plot=False)


# <<<<<<<<<<<<<<<<<<< Calibration characteristic traces  >>>>>>>>>>>>>>>>>>
'''Shift calibration characteristic traces'''
tar_ave_trace, tar_ave_trace_stdp, tar_ave_trace_stdm = targetTraces.average_trace(plot=False)
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
pn_combs, comp_cal_chars = TraceUtils.composite_char_traces(shifted_cal_chars, targetTraces.period, comp_num=composite_num)

# plt.figure(f'{composite_num}-composite char traces')
# for i, pn_tuple in enumerate(pn_combs):
#     if np.max(pn_tuple) <= 3:
#         plt.plot(comp_cal_chars[i], label=f'{pn_tuple}')
#
# <<<<<<<<<<<<<<<<<<< Test the composite search method  >>>>>>>>>>>>>>>>>>
target_data = targetTraces.get_data()

'''For some traces, find and plot the closest composite characteristic traces'''
test_num = 8
initial_trace = 3000
closest_k = 4  # half the number of composite char traces that will be identified
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

    idx_sort, diffs = sort_abs_volt_diff(trace, comp_cal_chars, k=closest_k)
    # idx_sort, diffs = sort_volt_diff(trace, comp_cal_chars, k=4)  # Here we are able to see why we need absolute value of voltage difference

    for idx in idx_sort:
        plt.plot(comp_cal_chars[idx], label=f'{pn_combs[idx]}')

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


# <<<<<<<<<<<<<<<<<<< Prepare canvas for plotting errors  >>>>>>>>>>>>>>>>>>
num_rows = (max_photon_number + 1)//2
num_cols = 2
fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 2*num_rows))
fig.canvas.manager.set_window_title('Average abs(voltage difference) from identified characteristic trace')

unique_pns = np.arange(max_photon_number+1)

# <<<<<<<<<<<<<<<<<<< Run minimum voltage difference method  >>>>>>>>>>>>>>>>>>
target_data = targetTraces.get_data()

print('Running minimum voltage diff method')

#TODO: how to speed this up? I tried parallelization with numba, but it didn't work very well.
'''Run a simple method: identify each trace with the closest comp char trace by smallest voltage diff'''
t1 = time.time()
pns, errors = search_smallest_diff(target_data, comp_cal_chars, pn_combs)
t2 = time.time()

print(f'Time for smallest voltage difference method is {t2-t1}')

plt.figure('minimum voltage difference bar')
plt.bar(list(range(max_photon_number + 1)), np.bincount(pns))
print(np.bincount(pns))
plt.ylim([0, 6000])

'''Plot errors'''
ave_errors = errors / targetTraces.period

for i, pn in enumerate(unique_pns):
    row = i // num_cols
    col = i % num_cols

    indices = np.where(pns == pn)
    axs[row, col].hist(ave_errors[indices], bins=100, alpha=0.5, label='Minimum voltage')

    axs[row, col].set_title(f'PN={pn}')
    axs[row, col].set_ylim([0, 300])


# <<<<<<<<<<<<<<<<<<< Run majority voting method  >>>>>>>>>>>>>>>>>>
target_data = targetTraces.get_data()

print('Running majority voting method')

#TODO: how to speed this up?
'''Run a simple majority voting method, where ties are settled by smallest area difference'''
t3 = time.time()
pns2, errors2 = search_maj_voting(target_data, comp_cal_chars, pn_combs, k=4)
t4 = time.time()

print(f'Time for majority voting method is {t4-t3}')

plt.figure('majority voting bar')
plt.bar(list(range(max_photon_number + 1)), np.bincount(pns2))
plt.ylim([0, 6000])

'''Plot the errors'''
ave_errors2 = errors2 / targetTraces.period

for i, pn in enumerate(unique_pns):
    row = i // num_cols
    col = i % num_cols

    indices2 = np.where(pns2==pn)
    axs[row, col].hist(ave_errors2[indices2], bins=100, alpha=0.5, label='Majority voting')

    axs[row, col].set_title(f'PN={pn}')
    axs[row, col].set_ylim([0, 300])


# # <<<<<<<<<<<<<<<<<<< Try to run 'batched' tail subtraction after identifying the photon numbers  >>>>>>>>>>>>>>>>>>
# # This doesn't work very well.
# subtract_data = subtract_tails_batch(target_data, pns, shifted_cal_chars, num_tails=composite_num-1)
# min_areaTraces = Traces(frequency, subtract_data, multiplier=multiplier, num_bins=num_bins)
# min_areaTraces.raw_histogram(plot=True, fig_name='raw hist minimum area difference method')
#
# subtract_data2 = subtract_tails_batch(target_data, pns2, shifted_cal_chars, num_tails=composite_num-1)
# maj_voteTraces = Traces(frequency, subtract_data2, multiplier=multiplier, num_bins=num_bins)
# maj_voteTraces.raw_histogram(plot=True, fig_name='raw hist majority voting method')
#

# <<<<<<<<<<<<<<<<<<< Compare with simple inner product stegosaurus method  >>>>>>>>>>>>>>>>>>
cal_errors = calibrationTraces.abs_voltage_diffs()

'''Plot the errors for calibration traces'''
for i, pn in enumerate(unique_pns):
    row = i // num_cols
    col = i % num_cols

    cal_ave_errors = cal_errors[pn] / calibrationTraces.period
    axs[row, col].hist(cal_ave_errors, bins=100, alpha=0.5, label='Calibration 100kHz')

    axs[row, col].set_title(f'PN={pn}')
    axs[row, col].set_ylim([0, 300])
    axs[row, col].legend()

plt.tight_layout()