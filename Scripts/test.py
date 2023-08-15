import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from src.ML_funcs import ML, return_artifical_data
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from tqdm.auto import tqdm
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import pywt
import pandas as pd
power = 8
frequency = 500
data, labels = return_artifical_data(frequency,1.5,power)
peak_data = []
def extract_features(x):
    peaks, props = find_peaks(x)
    peak_heights = x[peaks]
    if len(peaks)==0:
        peak_loc, max_peak = np.argmax(x), max(x)
    if len(peaks)>0:
        peak_loc, max_peak = peaks[np.argmax(peak_heights)], max(peak_heights)
    average = np.mean(x)
    std = np.std(x)
    y = np.argwhere(x ==max_peak/2)
    if len(y) ==0:
        rise_time = peak_loc/2
    if len(y)>0:
        rise_time = np.abs(y[0][0]-peak_loc)
    # if type(rise_time) != np.float64:
    #     print(type(rise_time))
    #     print(len(y))
    energy = np.sum(x**2)
    frequencies, psd = welch(x)
    if len(frequencies)==0:
        freq = 0
    if len(frequencies) >0:
        freq = frequencies[np.argmax(psd)]
    # grad =
    # max_grad = max(grad)
    # av_grad = np.mean(grad)

    return [peak_loc, average,std, energy, freq, max_peak, rise_time]

for series in tqdm(data):
    feature = extract_features(series)
    peak_data.append(feature)

features = np.array(peak_data)

model = ML(features, labels, modeltype='RF')
model.makemodel()
print(model.accuracy_score())

selector = SelectKBest(score_func=mutual_info_classif, k=3)
X_train, X_test, y_train, y_test = train_test_split(features, labels)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
important_feature_ind = selector.get_support(indices=True)
print(important_feature_ind)
actual_data = DataUtils.read_high_freq_data(frequency,power,new = True)
actual_trace = Traces(frequency,actual_data, 1.8,1000)
offset, _ = actual_trace.subtract_offset()
actual_data = actual_data - offset
extracted_features = []
for time_series in tqdm(actual_data):
    features = extract_features(time_series)
    extracted_features.append(features)
test = np.array(extracted_features)
predictions = model.predict(test)
y = np.bincount(predictions)
x = range(len(y))
plt.bar(x,y)

plt.show()

