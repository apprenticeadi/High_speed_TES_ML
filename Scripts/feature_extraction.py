import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from src.ML_funcs import ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from tsfresh import extract_features
#from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
from joblib import dump
from tqdm.auto import tqdm

# if __name__ == '__main__':
#     multiplier = 0.3
#     num_bins = 1000
#     guess_peak = 30
#     pca_components = 2  # it's really doubtful if pca helps at all
#     pca_cleanup = True
#
#     # <<<<<<<<<<<<<<<<<<< Calibation data  >>>>>>>>>>>>>>>>>>
#     data_100 = DataUtils.read_raw_data(100)
#     calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
#     _ = calibrationTraces.subtract_offset()
#     '''
#     create labelled dataset
#     '''
#     labels = calibrationTraces.return_labelled_traces()
#     filtered_ind = np.where(labels == -1)[0]
#     filtered_traces = np.delete(data_100, filtered_ind, axis = 0)
#     filtered_label = np.delete(labels, filtered_ind)
#     filtered_data = Traces(100, filtered_traces)
#     print(len(filtered_traces))
#     frequency = 500
#     data_high = filtered_data.overlap_to_high_freq(frequency)
#     print('dataset created')
#     filtered_label = filtered_label
#     data_high = data_high
#     df = pd.DataFrame(data_high, columns=[f'time_step_{i}' for i in range(100)])
#     df['id'] = range(len(df))
#     #ddf = dd.from_pandas(df, npartitions=10)
#     print('begin feature extraction')
#     extracted_features = extract_features(df, column_id='id')
#     print('done')
#     imputed_features = impute(extracted_features)
#     print('building model')
#     x = np.array(imputed_features)
#     X_train, X_test, y_train, y_test = train_test_split(x, filtered_label, test_size=0.2, random_state=42)
#
#     # Create a random forest classifier
#     classifier = RandomForestClassifier()
#
#     # Train the classifier
#     classifier.fit(X_train, y_train)
#     print('saving model')
#     model_filename = 'trained_500.joblib'
#     dump(classifier, model_filename)
#     # Predict the class labels for the test set
#     y_pred = classifier.predict(X_test)
#
#     # Calculate the accuracy of the classifier
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy:", accuracy)
#     actual_data = DataUtils.read_high_freq_data(frequency)
#     targetTraces = Traces(frequency=frequency, data=actual_data, multiplier=multiplier, num_bins=num_bins)
#     offset_target, _ = targetTraces.subtract_offset()
#     actual_data = actual_data - offset_target
#     test = classifier.predict((actual_data))
#     plt.bar(list(10),np.bincount(test))

from sktime.classification.kernel_based import RocketClassifier

rocket = RocketClassifier(num_kernels=2000)

multiplier = 0.3
num_bins = 1000
guess_peak = 30
pca_components = 2  # it's really doubtful if pca helps at all
pca_cleanup = True

multiplier = 0.3
num_bins = 1000
guess_peak = 30
pca_components = 2  # it's really doubtful if pca helps at all
pca_cleanup = True

# <<<<<<<<<<<<<<<<<<< Calibation data  >>>>>>>>>>>>>>>>>>
data_100 = DataUtils.read_raw_data(100)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
_ = calibrationTraces.subtract_offset()
'''
create labelled dataset
'''
labels = calibrationTraces.return_labelled_traces()
filtered_ind = np.where(labels == -1)[0]
filtered_traces = np.delete(data_100, filtered_ind, axis = 0)
filtered_label = np.delete(labels, filtered_ind)
filtered_data = Traces(100, filtered_traces)
print(len(filtered_traces))
frequency = 500
data_high = filtered_data.overlap_to_high_freq(frequency)
print('dataset created')

x_train, x_test, y_train, y_test = train_test_split(data_high, filtered_label, test_size=0.2, random_state=42)

rocket.fit(x_train, y_train)
print(accuracy_score(x_test, y_test))

