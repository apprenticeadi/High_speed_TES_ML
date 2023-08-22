import pickle
from sklearn.model_selection import train_test_split
from src.utils import DataUtils
from src.traces import Traces
import numpy as np
import sklearn.metrics as skm
import torch
from tsai.all import *
from src.utils import DataUtils, TraceUtils
import matplotlib.pyplot as plt
from src.ML_funcs import find_offset


frequency = 600
multiplier = 1.5
power = 8

learn = load_learner('./models/600kHz_RNN.pkl')

actual_data = DataUtils.read_high_freq_data(frequency,power = power,new = True)
shift = find_offset(frequency, power)
actual_data = actual_data - shift


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

