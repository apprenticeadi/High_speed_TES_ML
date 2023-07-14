import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from src.ML_funcs import ML

multiplier = 0.8
num_bins = 1000
guess_peak = 30
pca_components = 2  # it's really doubtful if pca helps at all
pca_cleanup = True

# <<<<<<<<<<<<<<<<<<< Calibation data  >>>>>>>>>>>>>>>>>>
data_100 = DataUtils.read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
# # need to put in shift here
# frequency = 900
# data_high = DataUtils.read_high_freq_data(frequency)  # unshifted
# targetTraces = Traces(frequency=frequency, data=data_high, multiplier=multiplier, num_bins=num_bins)
# tar_ave_trace, tar_ave_trace_stdp, tar_ave_trace_stdm = targetTraces.average_trace(plot=False)
# shifted_cal_traces = TraceUtils.shift_trace(tar_ave_trace, calibrationTraces, pad_length=guess_peak*2, id=1)


'''Shift data such that 0-photon trace has mean 0'''
_ = calibrationTraces.subtract_offset()

'''PCA cleanup calibration data'''
if pca_cleanup:
    data_cleaned = calibrationTraces.pca_cleanup(num_components=pca_components)
    calibrationTraces = Traces(frequency=100, data=data_cleaned, multiplier=multiplier, num_bins=num_bins)
    _ = calibrationTraces.subtract_offset()

'''
create labelled dataset
'''
labels = calibrationTraces.return_labelled_traces()
filtered_ind = np.where(labels == -1)[0]
filtered_traces = np.delete(data_100, filtered_ind, axis = 0)
filtered_label = np.delete(labels, filtered_ind)
print(len(filtered_label))

filtered_data = Traces(100, filtered_traces)

frequency = 500
freq_values =[ 500,600,700,800,900]
# fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
# plt.suptitle("Multiplier = 1", fontsize=14)
# for frequency, ax in zip(freq_values, axs.ravel()):
#     print(frequency)
#     data_high = filtered_data.overlap_to_high_freq(frequency)
#     model = ML(data_high, filtered_label, modeltype='C22')
#     model.makemodel(num_rounds=25)
#     actual_data = DataUtils.read_high_freq_data(frequency)
#     targetTraces = Traces(frequency=frequency, data=actual_data, multiplier=multiplier, num_bins=num_bins)
#     offset_target, _ = targetTraces.subtract_offset()
#     actual_data = actual_data - offset_target
#     print('data constructed')
#     if pca_cleanup:
#         actualTraces = Traces(frequency=frequency, data=actual_data)
#         actual_data = actualTraces.pca_cleanup(num_components=pca_components)
#
#     test = model.predict((actual_data))
#     #pn = test[100]
#     #indices = np.where(filtered_label == pn)[0]
#     #pn_data = data_high[indices]
#     #for x in pn_data:
#         #ax.plot(x,color = 'r', linestyle = 'dashed', linewidth = 0.1)
#     #for x in data_high:
#         #ax.plot(x,color = 'r', linestyle = 'dashed', linewidth = 0.2)
#     #ax.plot(actual_data[100],color = 'k', label = 'signal')
#     print('plotting')
#     ax.bar(list(range(len(np.bincount(test)))), np.bincount(test))
#     ax.set_title(str(frequency)+ 'kHz accuracy score = '+str(model.accuracy_score())[0:5])
#ax.set_title(str(frequency) + ' kHz, classifacation: ' + str(pn))
data_high = filtered_data.overlap_to_high_freq(frequency)
model = ML(data_high, filtered_label, modeltype='C22')
model.makemodel(num_rounds=25)
actual_data = DataUtils.read_high_freq_data(frequency)
targetTraces = Traces(frequency=frequency, data=actual_data, multiplier=multiplier, num_bins=num_bins)
offset_target, _ = targetTraces.subtract_offset()
actual_data = actual_data - offset_target
print('data constructed')
if pca_cleanup:
    actualTraces = Traces(frequency=frequency, data=actual_data)
    actual_data = actualTraces.pca_cleanup(num_components=pca_components)

test = model.predict((actual_data[0:200]))

print('plotting')
plt.bar(list(range(len(np.bincount(test)))), np.bincount(test))
plt.title(str(frequency)+ 'kHz accuracy score = '+str(model.accuracy_score())[0:5])
plt.show()