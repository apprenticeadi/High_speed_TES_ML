import numpy as np
from tqdm.auto import tqdm
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
data_100 = DataUtils.read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)

num_rows = 3
num_cols = 2
fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 2*num_rows))
fig.canvas.manager.set_window_title('photon number distribution for 5 frequencies')

freq_vals = [500,600,700,800,900]
for x in tqdm(freq_vals):
    frequency = x
    # data_high = calibrationTraces.overlap_to_high_freq(high_frequency=frequency)
    data_high = DataUtils.read_high_freq_data(frequency)
    targetTraces = Traces(frequency=frequency, data=data_high, multiplier=multiplier, num_bins=num_bins)
    freq_str = targetTraces.freq_str
    '''
    process calibration data to find range on traces for each photon number using total_traces
    '''
    total_traces = calibrationTraces.total_traces()
    max_photon_number = int((len(total_traces) / 3) - 1)
    cal_chars = calibrationTraces.characteristic_traces_pn(plot=False)
    max_photon_number = len(cal_chars) - 1
    '''
    apply shift
    '''
    tar_ave_trace, tar_ave_trace_stdp, tar_ave_trace_stdm = targetTraces.average_trace(plot=False)
    shifted_cal_chars = TraceUtils.shift_trace(tar_ave_trace, total_traces, pad_length=guess_peak * 2, id=1)
    shifted_cal_chars_old = TraceUtils.shift_trace(tar_ave_trace, cal_chars, pad_length=guess_peak * 2, id=1)
    '''
    generate composite characteristic traces, using composite_char_traces method
    '''
    pn_combs_old, comp_cal_chars_old = TraceUtils.composite_char_traces(shifted_cal_chars_old, targetTraces.period,
                                                                comp_num=composite_num)
    per = len(targetTraces.get_data()[0])
    pn_combs, comp_traces = TraceUtils.max_min_trace_utils(shifted_cal_chars, per)
    '''
    use search methods
    '''
    target_data = targetTraces.get_data()
    pns, errors = search_smallest_diff(target_data, comp_traces, pn_combs)
    pns_old, errors_old = search_smallest_diff(target_data, comp_cal_chars_old, pn_combs_old)
    row, col = 0, 0
    if x == 500: row, col = 0,0
    if x == 600: row, col = 0,1
    if x == 700: row, col = 1,0
    if x == 800: row, col = 1,1
    if x == 900: row, col = 2,0

    axs[row,col].bar(np.array(list(range(len(np.bincount(pns)))))-0.2, np.bincount(pns),width = 0.2, label = 'new method')
    axs[row,col].bar(np.array(list(range(len(np.bincount(pns_old)))))+0.2, np.bincount(pns_old),width = 0.2, label='old method')
    axs[row,col].set_title(str(x)+'kHz')
    axs[row, col].legend()
plt.tight_layout()
plt.show()