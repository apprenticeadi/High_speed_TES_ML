from sklearn.model_selection import train_test_split
from src.utils import DataUtils
from src.traces import Traces
import numpy as np
import sklearn.metrics as skm
import torch
from tsai.all import *
import time
import pickle


multiplier = 0.6
num_bins = 1000
guess_peak = 30
pca_components = 2  # it's really doubtful if pca helps at all
pca_cleanup = True

# <<<<<<<<<<<<<<<<<<< Calibation data  >>>>>>>>>>>>>>>>>>
data_100 = DataUtils.read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
_ = calibrationTraces.subtract_offset()
labels = calibrationTraces.return_labelled_traces()
filtered_ind = np.where(labels == -1)[0]
filtered_traces = np.delete(data_100, filtered_ind, axis = 0)
filtered_label = np.delete(labels, filtered_ind)

frequency = 500
filtered_data = Traces(100, filtered_traces)
data_high = filtered_data.overlap_to_high_freq(frequency)

time_series = data_high
labels = filtered_label


X_train, X_test, y_train, y_test = train_test_split(time_series, labels, test_size=0.2, random_state=42)


X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])
print(X.shape)

tfms = [None, TSClassification()]
batch_tfms = TSStandardize()
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms = batch_tfms, bs=[64,128])
#dls.show_batch(sharey = True)
print('data processed')
s = time.time()
model = build_ts_model(InceptionTimePlus, dls = dls)
learn = Learner(dls, model, metrics = accuracy)
f1 = time.time()
print('model built, time to build : ' + str(f1-s))
learn.lr_find()
learn.fit_one_cycle(10, lr_max = 0.001)
PATH = Path('./models/500kHz.pkl')
PATH.parent.mkdir(parents = True, exist_ok = True)
learn.export(PATH)
