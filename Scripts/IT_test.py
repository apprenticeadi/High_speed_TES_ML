import pickle
from sklearn.model_selection import train_test_split
from src.utils import DataUtils
from src.traces import Traces
import numpy as np
import sklearn.metrics as skm
import torch
from tsai.all import *
from src.utils import DataUtils
import matplotlib.pyplot as plt


frequency = 500
multiplier = 0.6
num_bins = 1000

learn = load_learner('./models/500kHz.pkl')
actual_data = DataUtils.read_high_freq_data(frequency)
targetTraces = Traces(frequency=frequency, data=actual_data, multiplier=multiplier, num_bins=num_bins)
offset_target, _ = targetTraces.subtract_offset()
actual_data = actual_data - offset_target

dls = learn.dls
valid_dl = dls.valid

y_fake = np.zeros(len(actual_data))
X, y, splits = combine_split_data([actual_data], [y_fake])

test_ds = dls.dataset.add_test(X,y)
test_dl = valid_dl.new(test_ds)
print('data and model loaded')

probas, targets, preds = learn.get_preds(dl = test_dl, with_decoded = True, save_preds = None, save_targs = None)
print('predictions made')
print('plotting')
plt.bar(list(range(len(np.bincount(preds)))), np.bincount(preds))
plt.title('PN distribution for ' +str(frequency)+' kHz'+ ' and m=' +str(multiplier))
plt.show()
