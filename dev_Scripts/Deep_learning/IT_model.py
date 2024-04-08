import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sktime.transformations.panel.rocket import Rocket
from src.utils import DataUtils
from src.traces import Traces
import numpy as np
import sklearn.metrics as skm
import torch
from tsai.all import *
import time
import pickle
from src.ML_funcs import return_artifical_data

'''
process training data
'''
frequency = 600
power = 8
modeltype = RNN


time_series, labels = return_artifical_data(frequency,1.5,power)
X_train, X_test, y_train, y_test = train_test_split(time_series, labels, test_size=0.2, random_state=42)
X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])


tfms = [None, TSClassification()]
batch_tfms = TSStandardize()
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms = batch_tfms, bs=[64])

'''
build the model, argument of build_ts_model specifies the model. Available models found on : https://timeseriesai.github.io/tsai/
most used models : RNN, recurrent neural network
                   InceptionTime, state of the art method
                   MultiRocketPlus, state of the art
                   ....
                   
'''

model = build_ts_model(modeltype, dls = dls)
learn = Learner(dls, model, metrics = accuracy)

'''
fit and save the model
'''
learn.lr_find()
learn.fit_one_cycle(200, lr_max = 0.001)
PATH = Path(f'./models/{frequency}kHz_{modeltype}.pkl')
PATH.parent.mkdir(parents = True, exist_ok = True)
learn.export(PATH)
'''
print out metrics
'''
interp = ClassificationInterpretation.from_learner(learn)
interp.print_classification_report()
