import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from sklearn.model_selection import train_test_split


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
offset_cal, _ = calibrationTraces.subtract_offset()

#target data - 'read_high_freq_data' = actual data, uncomment to use
frequency = 600
data_high = calibrationTraces.overlap_to_high_freq(high_frequency=frequency)
# data_high = DataUtils.read_high_freq_data(frequency)
targetTraces = Traces(frequency=frequency, data=data_high, multiplier=multiplier, num_bins=num_bins)
freq_str = targetTraces.freq_str
#
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
create a labelled dataset for training/ testing, labelled_comp_traces is a list of all traces with photon number as index
'''
labelled_comp_traces = []
labelled_pn_combs = []

for i in range(10):
    indices = np.arange(i,3000,10)
    new_array = comp_traces[indices]
    new_arry1 = pn_combs[indices]
    labelled_comp_traces.append(new_array)
    labelled_pn_combs.append((new_arry1))


'''
creating dataset
'''

dataset = np.concatenate((labelled_comp_traces[0], labelled_comp_traces[1], labelled_comp_traces[2],
                          labelled_comp_traces[3],labelled_comp_traces[4],labelled_comp_traces[5],labelled_comp_traces[6],
                          labelled_comp_traces[7],labelled_comp_traces[8],labelled_comp_traces[9]))

num = len(labelled_comp_traces[0])

labels = np.array([0]*num + [1]*num + [2]*num + [3]*num + [4]*num + [5]*num +
                  [6]*num + [7]*num + [8]*num + [9]*num)
x_train ,  x_test, y_train, y_test = train_test_split(dataset, labels)


dtrain = xgb.DMatrix(x_train, label = y_train)
dtest = xgb.DMatrix(x_test, label = y_test)

params = {
    'max_depth':3,
    'eta' : 0.1,
    'num_class' : 10,
    'objective': 'multi:softmax'
}

num_rounds = 10
model = xgb.train(params, dtrain, num_rounds)

predictions = model.predict(dtest)
print(type(predictions))