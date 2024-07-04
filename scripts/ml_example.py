import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

from tes_resolver import Traces, DataChopper, config, generate_training_traces
from tes_resolver.classifier import InnerProductClassifier, TabularClassifier, CNNClassifier
from utils import DFUtils, DataReader

'''Run ml classifier to classify all the data in a certain folder. '''
# parameters
cal_rep_rate = 100  # the rep rate to generate training
high_rep_rate = 800  # the higher rep rates to predict

modeltype = 'KNN'  # machine learning model
test_size = 0.1  # machine learning test-train split ratio

# read data
sampling_rate = 5e4
dataReader = DataReader('Data/Tomography_data_2024_04}')
data_group = 'power_6'

# read calibration data
cal_data = dataReader.read_raw_data(data_group, cal_rep_rate)
calTraces = Traces(cal_rep_rate, cal_data, parse_data=True, trigger_delay=0)

# Train an ip classifier, and use it to label the calibration data
ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)
ipClassifier.train(calTraces)
ipClassifier.predict(calTraces, update=True)

# Remove the baseline for calibration traces
cal_baseline = calTraces.find_offset()
calTraces.data = calTraces.data - cal_baseline  # remove the baseline

# Load actual traces
actual_data = dataReader.read_raw_data(data_group, high_rep_rate)

if high_rep_rate <= 300:  # set suitable trigger delay
    trigger_delay = 0
else:
    trigger_delay = DataChopper.find_trigger(actual_data, samples_per_trace=int(sampling_rate / high_rep_rate))
actualTraces = Traces(high_rep_rate, actual_data, parse_data=True, trigger_delay=trigger_delay)

# Generate training
trainingTraces = generate_training_traces(calTraces, high_rep_rate, trigger_delay=trigger_delay)

# correct for the vertical shift
offset = np.max(trainingTraces.average_trace()) - np.max(actualTraces.average_trace())
trainingTraces.data = trainingTraces.data - offset

# Train ML Classifier
print(f'Training ml classifier for {high_rep_rate}kHz')
t1 = time.time()
mlClassifier = TabularClassifier(modeltype, test_size=test_size)
mlClassifier.train(trainingTraces)
t2 = time.time()
accuracy = mlClassifier.accuracy_score

print(f'Training finished after {t2 - t1}s. Accuracy score = {accuracy}. ')

# Predict unknown traces
print(f'Making predictions for {actualTraces.num_traces} traces')
t3 = time.time()
mlClassifier.predict(actualTraces, update=True)
t4 = time.time()
print(f'Prediction finished after {t4 - t3}s. ')

# Plot results
plt.figure('Photon number distribution', layout='constrained')
pns, cal_distrib = calTraces.pn_distribution(normalised=True)  # calibration data distribution
plt.bar(pns, cal_distrib, label='Calibration')

pn_labels, predicted_distrib = actualTraces.pn_distribution(normalised=True)
plt.bar(pn_labels, predicted_distrib, label='Unknown')  # predicted distribution

plt.xlabel('Photon number')
plt.ylabel('Probability')


