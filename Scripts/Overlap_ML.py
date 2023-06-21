import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils
from src.traces import Traces
from src.ML_funcs import ML

multiplier = 0.5
num_bins = 1000
guess_peak = 30
pca_components = 1  # it's really doubtful if pca helps at all
composite_num = 4

# <<<<<<<<<<<<<<<<<<< Calibation data  >>>>>>>>>>>>>>>>>>
data_100 = DataUtils.read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)

'''Shift data such that 0-photon trace has mean 0'''
offset_cal, _ = calibrationTraces.subtract_offset()

'''
create labelled dataset
'''
ld = calibrationTraces.return_labelled_traces()

labels = []
for i in range(10):
    nums = [i]*len(ld[i])
    labels.append(nums)

flat_ld = [item for sublist in ld for item in sublist]
flat_labels = [item for sublist in labels for item in sublist]

'''
overlap to high frequency traces
'''

filtered_data = Traces(100, np.array(flat_ld))
#frequency = 500
freq_values =[ 500,600,700,800,900]
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))

for frequency, ax in zip(freq_values, axs.ravel()):
    data_high = filtered_data.overlap_to_high_freq(frequency)
    model = ML(data_high, flat_labels, modeltype='BDT')
    model.makemodel(num_rounds=20)
    actual_data = DataUtils.read_high_freq_data(frequency)
    test = model.predict((actual_data))
    ax.bar(list(range(len(np.bincount(test)))), np.bincount(test))
    ax.set_title(str(frequency)+ 'kHz accuracy score = '+str(model.accuracy_score())[0:5])
plt.show()


