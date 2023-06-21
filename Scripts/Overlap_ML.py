import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils
from src.traces import Traces
from src.ML_funcs import ML

multiplier = 1
num_bins = 1000
guess_peak = 30
pca_components = 2  # it's really doubtful if pca helps at all
pca_cleanup = True

# <<<<<<<<<<<<<<<<<<< Calibation data  >>>>>>>>>>>>>>>>>>
data_100 = DataUtils.read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)

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


filtered_data = Traces(100, filtered_traces)

#frequency = 500
freq_values =[ 500,600,700,800,900]
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))

for frequency, ax in zip(freq_values, axs.ravel()):
    data_high = filtered_data.overlap_to_high_freq(frequency)
    model = ML(data_high, filtered_label, modeltype='RF')
    model.makemodel(num_rounds=25)
    actual_data = DataUtils.read_high_freq_data(frequency)

    if pca_cleanup:
        actualTraces = Traces(frequency=frequency, data=actual_data)
        actual_data = actualTraces.pca_cleanup(num_components=pca_components)

    test = model.predict((actual_data))
    ax.bar(list(range(len(np.bincount(test)))), np.bincount(test))
    ax.set_title(str(frequency)+ 'kHz accuracy score = '+str(model.accuracy_score())[0:5])
plt.show()


