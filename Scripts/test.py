import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from sklearn.feature_selection import SelectKBest,mutual_info_classif, f_classif
from src.ML_funcs import ML, return_artifical_data, extract_features
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from tqdm.auto import tqdm
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import pandas as pd
power = 6
frequency = 400
data, labels = return_artifical_data(frequency,1.5,power)
peak_data = []
'''
calculate features for each time-series
'''
for series in tqdm(data):
    feature = extract_features(series)
    peak_data.append(feature)

features = np.array(peak_data)
'''
train model on time series
'''
model = ML(features, labels, modeltype='RF')
model.makemodel()
print(model.accuracy_score())
'''
find most important features
'''
selector = SelectKBest(score_func=mutual_info_classif, k=9)
X_train, X_test, y_train, y_test = train_test_split(features, labels)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
important_feature_ind = selector.get_support(indices=True)
print("Indices of Important Features:", important_feature_ind)

# Print F-scores of all features
all_feature_scores = selector.scores_
for idx, score in enumerate(all_feature_scores):
    print(f"Feature {idx}: Score = {score}")
'''
load in actual data
'''
actual_data = DataUtils.read_high_freq_data(frequency,power,new = True)
actual_trace = Traces(frequency,actual_data, 1.8,1000)
#offset, _ = actual_trace.subtract_offset()
actual_data = actual_data - 2000
'''
extract features for actual data
'''
extracted_features = []
for time_series in tqdm(actual_data):
    features = extract_features(time_series)
    extracted_features.append(features)

'''
make predicitons
'''
test = np.array(extracted_features)
predictions = model.predict(test)
y = np.bincount(predictions)
x = range(len(y))

plt.bar(x,y)
plt.show()

