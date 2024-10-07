import numpy as np
import time
import os
import pandas as pd
import logging

from tes_resolver import Traces, DataChopper, config, generate_training_traces
from tes_resolver.classifier import InnerProductClassifier, TabularClassifier, CNNClassifier
from utils import DFUtils, DataReader, RuquReader, LogUtils

'''Run ml classifier to classify all the data in a certain folder. '''
# parameters
cal_rep_rate = 100  # the rep rate to generate training
cal_date = '2024-07-17-1954'

high_rep_rate = 800 # np.arange(100, 1100, 100)  # the higher rep rates to predict
high_date = '2024-07-17-2010'

sampling_rate = 5e4
data_name = r'squeezed states 2024_07_17'
dataReader = RuquReader(rf'Data\{data_name}')
data_groups = ['Chan[1]', 'Chan[2]']

modeltype = 'KNN'  # machine learning model
test_size = 0.1  # machine learning test-train split ratio

results_dir = os.path.join(config.home_dir, '..', 'Results', data_name, modeltype,
                               f'sq_data_{config.time_stamp}')

# logging
LogUtils.log_config(config.time_stamp, results_dir, 'log')
logging.info(f'Processing squeezed data. Calibration data at {cal_rep_rate}kHz collected on {cal_date}. '
             f'Traget data at {high_rep_rate}kHz collected on {high_date}. ML used on calibration data as well,'
             f'which is split into half for training and half for prediction.')

for data_group in data_groups:
    print(f'\nProcessing {data_group}...')

    # read calibration data
    # cal_data = dataReader.read_raw_data(data_group, cal_rep_rate)
    cal_data = dataReader.read_raw_data(f'{cal_rep_rate}kHz', data_group, cal_date, concatenate=True, return_file_names=False)
    calTraces = Traces(cal_rep_rate, cal_data, parse_data=True, trigger_delay=0)

    # Train an ip classifier, and use it to label the calibration data
    ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)

    t1 = time.time()
    ipClassifier.train(calTraces)
    t2 = time.time()
    ipClassifier.predict(calTraces, update=True)
    t3 = time.time()
    print(
        f'For {cal_rep_rate}kHz, ip classifier trained after {t2 - t1}s, predict {calTraces.num_traces} traces after {t3 - t2}s.')

    pns, cal_distrib = calTraces.pn_distribution(normalised=True)
    print(f'PN distribution is {cal_distrib}')

    results_df = pd.DataFrame(columns=['rep_rate', 'num_traces', 'acc_score', 'training_t', 'predict_t'] + list(pns))
    results_df.to_csv(DFUtils.create_filename(results_dir + rf'\{modeltype}_results_{data_group}.csv'), index=False)

    # Remove the baseline for calibration traces
    cal_baseline = calTraces.find_offset()
    calTraces.data = calTraces.data - cal_baseline  # remove the baseline

    # ML for higher rep rates
    for i_rep, rep_rate in enumerate([cal_rep_rate, high_rep_rate]):
        print('')

        if rep_rate == cal_rep_rate:
            # use half the 100kHz data to train classifier, classifier to predict the rest
            training_data = calTraces.data[:calTraces.num_traces // 2]
            training_labels = calTraces.labels[:len(training_data)]
            trainingTraces = Traces(rep_rate, training_data, labels=training_labels, parse_data=False)

            actual_data = calTraces.data[calTraces.num_traces // 2:]
            actualTraces = Traces(rep_rate, actual_data, parse_data=False)

            print(
                f'Load first half of {rep_rate}kHz traces as training data, '
                f'use classifier to predict second half')

        else:
            # Load actual traces for non-100kHz data
            ti = time.time()
            actual_data = dataReader.read_raw_data(f'{rep_rate}kHz', data_group, high_date, concatenate=True, return_file_names=False)
            actualTraces = Traces(rep_rate, actual_data, parse_data=True, trigger_delay='automatic')
            trigger_delay = actualTraces.trigger_delay
            tf = time.time()
            print(f'Load high rep rate data into traces took {tf - ti}s')

            # Generate training
            ti = time.time()
            trainingTraces = generate_training_traces(calTraces, rep_rate, trigger_delay=trigger_delay)
            # correct for the vertical shift
            offset = np.max(trainingTraces.average_trace()) - np.max(actualTraces.average_trace())
            trainingTraces.data = trainingTraces.data - offset
            tf = time.time()
            print(f'Generate training traces took {tf - ti}s')

        # ML Classifier
        print(f'Training ml classifier for {rep_rate}kHz')
        t1 = time.time()
        mlClassifier = TabularClassifier(modeltype, test_size=test_size)
        mlClassifier.train(trainingTraces)
        t2 = time.time()

        accuracy = mlClassifier.accuracy_score

        print(f'Making predictions for {actualTraces.num_traces} traces')
        t3 = time.time()
        mlClassifier.predict(actualTraces, update=True)
        t4 = time.time()
        print(
            f'Training finished after {t2 - t1}s. Accuracy score = {accuracy}. Prediction finished after {t4 - t3}s. ')

        # results
        raw_labels = actualTraces.labels
        np.savetxt(DFUtils.create_filename(results_dir + rf'\{modeltype}_pn_labels_{rep_rate}kHz_{data_group}.txt'),
                   raw_labels)

        pn_labels, predicted_distrib = actualTraces.pn_distribution(normalised=True)
        yvals = np.zeros_like(cal_distrib)
        yvals[:len(predicted_distrib)] = predicted_distrib
        print(f'Predicted pn distribution = {yvals}')

        # Save results
        results_df.loc[i_rep] = [rep_rate, actualTraces.num_traces, accuracy, t2 - t1, t4 - t3] + list(yvals)
        results_df.to_csv(DFUtils.create_filename(results_dir + rf'\{modeltype}_results_{data_group}.csv'), index=False)


