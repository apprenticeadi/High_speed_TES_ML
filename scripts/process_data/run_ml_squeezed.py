import numpy as np
import time
import os
import pandas as pd

from tes_resolver import Traces, DataChopper, config, generate_training_traces
from tes_resolver.classifier import InnerProductClassifier, TabularClassifier, CNNClassifier
from utils import DFUtils, DataReader, RuquReader

'''Run ml classifier to classify all the data in a certain folder. '''
# parameters
cal_rep_rate = 100  # the rep rate to generate training
high_rep_rates = [100, 800] # np.arange(100, 1100, 100)  # the higher rep rates to predict

modeltype = 'KNN'  # machine learning model
test_size = 0.1  # machine learning test-train split ratio

# read data
sampling_rate = 5e4
data_name = r'squeezed states 2024_07_17'
cal_name = 'coh_states_2024_07_11'
# dataReader = DataReader(f'Data/{data_name}')
# powers = np.arange(0, 12)
# data_groups = np.array([f'power_{p}' for p in powers])

dataReader = RuquReader(rf'Data\{data_name}')
calReader = RuquReader(rf'Data\{cal_name}')
data_groups = ['Chan[1]', 'Chan[2]']
date_keywords = ['2024-07-17-1954', '2024-07-17-2010']
cal_keyword = 'power_1'

update_params = False  # whether to save the results in a params folder
if update_params:
    params_dir = os.path.join(config.home_dir, '..', 'Results', data_name, 'Params', modeltype)

for data_group in data_groups:
    print(f'\nProcessing {data_group}...')

    # read calibration data
    # cal_data = dataReader.read_raw_data(data_group, cal_rep_rate)
    cal_data = calReader.read_raw_data(f'{cal_rep_rate}kHz', data_group, cal_keyword, concatenate=True, return_file_names=False)
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

    results_dir = os.path.join(config.home_dir, '..', 'Results', data_name, modeltype,
                               f'{cal_name}_trained_{date_keywords}_data_{config.time_stamp}')
    results_df = pd.DataFrame(columns=['rep_rate', 'num_traces', 'acc_score', 'training_t', 'predict_t'] + list(pns))
    results_df.to_csv(DFUtils.create_filename(results_dir + rf'\{modeltype}_results_{data_group}.csv'), index=False)

    # Remove the baseline for calibration traces
    cal_baseline = calTraces.find_offset()
    calTraces.data = calTraces.data - cal_baseline  # remove the baseline

    np.savetxt(results_dir + rf'\cal_data_{data_group}.txt', calTraces.data)

    # ML for higher rep rates
    for i_rep, high_rep_rate in enumerate(high_rep_rates):
        print('')

        # file to save classifier
        filedir = results_dir + r'\saved_classifiers'
        filename = rf'{modeltype}_trained_by_{data_group}_{high_rep_rate}kHz'

        if high_rep_rate == cal_rep_rate:
            # use half the 100kHz data to train classifier, classifier to predict the rest
            training_data = calTraces.data[:calTraces.num_traces // 2]
            training_labels = calTraces.labels[:len(training_data)]
            trainingTraces = Traces(high_rep_rate, training_data, labels=training_labels, parse_data=False)

            actual_data = calTraces.data[calTraces.num_traces // 2:]
            actualTraces = Traces(high_rep_rate, actual_data, parse_data=False)

            print(
                f'Load first half of {high_rep_rate}kHz traces as training data, use classifier to predict second half')

        else:
            # Load actual traces
            ti = time.time()
            # actual_data = dataReader.read_raw_data(data_group, high_rep_rate)
            actual_data = dataReader.read_raw_data(f'{high_rep_rate}kHz', data_group, date_keywords[i_rep], concatenate=True, return_file_names=False)
            actualTraces = Traces(high_rep_rate, actual_data, parse_data=True, trigger_delay='automatic')
            trigger_delay = actualTraces.trigger_delay
            tf = time.time()
            print(f'Load high rep rate data into traces took {tf - ti}s')

            # Generate training
            ti = time.time()
            trainingTraces = generate_training_traces(calTraces, high_rep_rate, trigger_delay=trigger_delay)
            # correct for the vertical shift
            # offset = np.max(trainingTraces.average_trace()) - np.max(actualTraces.average_trace())
            # trainingTraces.data = trainingTraces.data - offset
            tf = time.time()
            print(f'Generate training traces took {tf - ti}s')

        np.savetxt(results_dir + rf'\actual_data_{data_group}_{high_rep_rate}kHz.txt', actualTraces.data)
        np.savetxt(results_dir + rf'\training_data_{data_group}_{high_rep_rate}kHz.txt', trainingTraces.data)

        # ML Classifier
        print(f'Training ml classifier for {high_rep_rate}kHz')
        t1 = time.time()
        if modeltype == 'CNN':
            mlClassifier = CNNClassifier(test_size=test_size)
            mlClassifier.train(trainingTraces, checkpoint_file=os.path.join(filedir, filename + '_checkpoint'))
        else:
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
        np.savetxt(DFUtils.create_filename(results_dir + rf'\{modeltype}_pn_labels_{high_rep_rate}kHz_{data_group}.txt'), raw_labels)

        pn_labels, predicted_distrib = actualTraces.pn_distribution(normalised=True)
        yvals = np.zeros_like(cal_distrib)
        yvals[:len(predicted_distrib)] = predicted_distrib
        print(f'Predicted pn distribution = {yvals}')

        # Save results
        results_df.loc[i_rep] = [high_rep_rate, actualTraces.num_traces, accuracy, t2 - t1, t4 - t3] + list(yvals)
        results_df.to_csv(DFUtils.create_filename(results_dir + rf'\{modeltype}_results_{data_group}.csv'), index=False)

        # mlClassifier.save(filename=filename, filedir=filedir)

    # update params folder
    if update_params:
        results_df.to_csv(DFUtils.create_filename(params_dir + rf'\{modeltype}_results_{data_group}.csv'), index=False)
