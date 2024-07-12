import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

from tes_resolver import Traces, DataChopper, config, generate_training_traces
from tes_resolver.classifier import InnerProductClassifier, TabularClassifier, CNNClassifier
from utils import DFUtils, tvd, DataReader, RuquReader

'''Run ml classifier to classify all the data in a certain folder. '''
# parameters
cal_rep_rate = 100  # the rep rate to generate training
high_rep_rate = 800  # the higher rep rates to predict
sampling_rate = 5e4
data_keywords = ['1.75nmPump', '1570nmBPF', 'Chan[1]']

modeltype = 'KNN'  # machine learning model
test_size = 0.1  # machine learning test-train split ratio
plot_training = True  # whether to plot the calibration data and how training traces is generated
traces_to_plot = 100
# read data
sqReader = RuquReader(r'Data\TES data backup 20240711\squeezed states 2024_07_11')

# read calibration data
cal_data = sqReader.read_raw_data(f'{cal_rep_rate}kHz', '112uW', '-1859_', *data_keywords, concatenate=True)
calTraces = Traces(cal_rep_rate, cal_data, parse_data=True, trigger_delay=0)

# Load actual traces
actual_data = sqReader.read_raw_data(f'{high_rep_rate}kHz', '900uW', '-1842_', *data_keywords, concatenate=False)[:4]
actual_data = np.concatenate(actual_data)
actualTraces = Traces(high_rep_rate, actual_data, parse_data=True, trigger_delay= 'automatic')

# remove saturating traces, if necessary
sat_indices = np.where(np.max(actualTraces.data, axis=1) > 30000)[0]
parsed_data = np.delete(actualTraces.data, sat_indices, axis=0)
actualTraces.data = parsed_data

# Train an ip classifier, and use it to label the calibration data
ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)
ipClassifier.train(calTraces)
ipClassifier.predict(calTraces, update=True)

# Remove the baseline for calibration traces
cal_baseline = calTraces.find_offset()
calTraces.data = calTraces.data - cal_baseline  # remove the baseline

# Generate training
trainingTraces = generate_training_traces(calTraces, high_rep_rate, trigger_delay= actualTraces.trigger_delay)

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

print(f'Training finished after {t2 - t1}s for {trainingTraces.num_traces} training traces. Accuracy score = {accuracy}. ')

# Predict unknown traces
print(f'Making predictions for {actualTraces.num_traces} traces')
t3 = time.time()
mlClassifier.predict(actualTraces, update=True)
t4 = time.time()
print(f'Prediction finished after {t4 - t3}s. ')

# Plot results
plt.figure('Photon number distribution', layout='constrained')
pns, cal_distrib = calTraces.pn_distribution(normalised=True)  # calibration data distribution
plt.bar(pns-0.2, cal_distrib, width=0.4, label=f'{cal_rep_rate}kHz, av={np.mean(calTraces.labels):.2g}', alpha=0.8)

pn_labels, predicted_distrib = actualTraces.pn_distribution(normalised=True)
plt.bar(pn_labels+0.2, predicted_distrib, width=0.4, label=f'{high_rep_rate}kHz, av={np.mean(actualTraces.labels):.2g}', alpha=0.8)  # predicted distribution

plt.xlabel('Photon number')
plt.ylabel('Probability')
plt.legend()

# calulate the total variation distance
print(f'Total variation distance = {tvd(cal_distrib, predicted_distrib)}')

# plot the calibration data
if plot_training:
    fig, axs = plt.subplot_mosaic([[0, 0], [1, 2]], figsize=(12, 8), layout='constrained', sharey=True)
    ax=axs[0]
    for trace in range(traces_to_plot):
        ax.plot(calTraces.data[trace], alpha=0.05)
    char_traces_dict = calTraces.characteristic_traces()
    for pn in char_traces_dict.keys():
        ax.plot(char_traces_dict[pn], color='blue', alpha=0.5)

    ax.axhline(cal_baseline, color='black', linestyle='--', label='Baseline')
    ax.text(calTraces.period-10, cal_baseline*2, f'{cal_baseline:.2f}', color='black')
    ax.set_title(f'{cal_rep_rate}kHz calibration data')
    ax.set_xlabel(f'Samples at {sampling_rate/1000}MHz')
    ax.set_ylabel('Voltage bit value')

    # plot training traces
    ax=axs[1]
    for trace in range(traces_to_plot):
        ax.plot(trainingTraces.data[trace], alpha=0.05)
    ax.plot(trainingTraces.average_trace(), alpha=0.5, color='red')
    av_peak_train = np.max(trainingTraces.average_trace())
    ax.axhline(av_peak_train, color='black', linestyle='--', label='Peak')
    ax.text(trainingTraces.period-10, av_peak_train*1.1, f'{av_peak_train:.2f}', color='black')
    ax.set_title(f'Training traces for {high_rep_rate}kHz')
    ax.set_ylabel('Voltage bit value')
    ax.set_xlabel(f'Samples at {sampling_rate/1000}MHz')

    train_char_traces = trainingTraces.characteristic_traces()
    for pn in train_char_traces.keys():
        ax.plot(train_char_traces[pn], color='blue', alpha=0.5)

    ax.plot(trainingTraces.data[np.argmin(trainingTraces.data) // trainingTraces.period], color='green')

    # plot actual traces
    ax = axs[2]
    for trace in range(traces_to_plot):
        ax.plot(actualTraces.data[trace], alpha=0.05)
    ax.plot(actualTraces.average_trace(), alpha=0.5, color='red')
    av_peak = np.max(actualTraces.average_trace())
    ax.axhline(av_peak, color='black', linestyle='--', label='Peak')
    ax.text(actualTraces.period-10, av_peak*1.1, f'{av_peak:.2f}', color='black')
    ax.set_title(f'Actual traces for {high_rep_rate}kHz')
    ax.set_xlabel(f'Samples at {sampling_rate/1000}MHz')

    predicted_char_traces = actualTraces.characteristic_traces()
    for pn in predicted_char_traces.keys():
        ax.plot(predicted_char_traces[pn], color='blue', alpha=0.5)

    ax.plot(actualTraces.data[np.argmin(actualTraces.data) // actualTraces.period], color='green')

plt.show()

