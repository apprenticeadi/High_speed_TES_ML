import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

from tes_resolver.ml_funcs import generate_training_traces
from tes_resolver.traces import Traces
from tes_resolver.classifier import InnerProductClassifier, TabularClassifier
from tes_resolver.data_chopper import DataChopper
from src.data_reader import DataReader
from src.utils import DFUtils
import tes_resolver.config as config

'''Parameters'''
cal_rep_rate = 100  # the rep rate to generate training
high_rep_rates = np.arange(200, 1100, 100)  # the higher rep rates to predict

# ML parameters
modeltype='RF'
test_size=0.1

# read data
sampling_rate = 5e4
dataReader = DataReader('Data/Tomography_data_2024_04')
powers = np.arange(12)
data_groups = np.array([f'power_{p}' for p in powers])

for data_group in data_groups:
    print(f'\nProcessing {data_group}...')
    # save data
    results_dir = os.path.join(config.home_dir, '..', 'Results', 'Tomography_data_2024_4', f'{modeltype}', f'{data_group}_{config.time_stamp}')
    # results_dir = rf'..\..\Results\Tomography_data_2024_04\{modeltype}\{data_group}_{config.time_stamp}'

    '''Read the calibration data'''
    cal_data = dataReader.read_raw_data(data_group, cal_rep_rate)
    calTraces = Traces(cal_rep_rate, cal_data, parse_data=True, trigger_delay=0)

    '''Train an ip classifier, and use it to label the calibration data '''
    ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)

    t1 = time.time()
    ipClassifier.train(calTraces)
    t2 = time.time()
    ipClassifier.predict(calTraces, update=True)
    t3 = time.time()
    print(f'For {cal_rep_rate}kHz, ip classifier trained after {t2-t1}s, predict {calTraces.num_traces} traces after {t3-t2}s.')

    pns, cal_distrib = calTraces.pn_distribution(normalised=True)
    print(f'PN distribution is {cal_distrib}')

    # Result file
    results_df = pd.DataFrame(columns=['rep_rate', 'num_traces', 'acc_score', 'training_t', 'predict_t'] + list(pns))
    # results_df.loc[0] = [cal_rep_rate, calTraces.num_traces, np.nan, t2-t1, t3-t2] + list(cal_distrib)
    results_df.to_csv(DFUtils.create_filename(results_dir + rf'\{modeltype}_results_{data_group}.csv'), index=False)

    '''Plot the calibration data stegosaurus'''
    # fig1, ax1 = plt.subplots(layout='constrained', figsize=(12,8))
    # overlaps = ipClassifier.calc_inner_prod(calTraces)
    # ax1.hist(overlaps, bins=ipClassifier.num_bins, alpha=1, label=data_group)
    # ip_bins = ipClassifier.inner_prod_bins
    # for pn in ip_bins.keys():
    #     ax1.axvline(ip_bins[pn], ymin=0, ymax=0.25, ls='dashed', color='black')
    # ax1.set_xlabel('Inner product')
    # ax1.set_ylabel('Occurences')
    # ax1.set_title(f'IPClassifier trained by {ip_training_group}')
    # plt.show()

    '''Remove the baseline for calibration traces'''
    cal_baseline = calTraces.find_offset()
    calTraces.data = calTraces.data - cal_baseline  # remove the baseline

    '''ML for higher rep rates'''
    for i_rep, high_rep_rate in enumerate(high_rep_rates):

        '''Load actual traces'''
        ti = time.time()
        actual_data = dataReader.read_raw_data(data_group, high_rep_rate)

        # set suitable trigger delay
        if high_rep_rate <= 300:
            trigger_delay = 0
        else:
            trigger_delay = DataChopper.find_trigger(actual_data, samples_per_trace=int(sampling_rate/high_rep_rate))

        actualTraces = Traces(high_rep_rate, actual_data, parse_data=True, trigger_delay=trigger_delay)
        tf = time.time()
        print(f'Load high rep rate data into traces took {tf-ti}s')

        '''Generate training'''
        ti = time.time()
        trainingTraces = generate_training_traces(calTraces, high_rep_rate, trigger_delay=trigger_delay)

        # correct for the vertical shift
        offset = np.max(trainingTraces.average_trace()) - np.max(actualTraces.average_trace())
        trainingTraces.data = trainingTraces.data - offset
        tf = time.time()
        print(f'Generate training traces took {tf-ti}s')

        '''ML Classifier'''
        t1 = time.time()
        mlClassifier = TabularClassifier(modeltype, test_size=test_size)

        print(f'\nTraining ml classifier for {high_rep_rate}kHz')
        mlClassifier.train(trainingTraces)
        t2 = time.time()

        accuracy = mlClassifier.accuracy_score

        print(f'Making predictions for {actualTraces.num_traces} traces')
        t3 = time.time()
        mlClassifier.predict(actualTraces, update=True)
        t4 = time.time()
        print(f'Training finished after {t2-t1}s. Accuracy score = {accuracy}. Prediction finished after {t4-t3}s. ')

        # results
        pn_labels, predicted_distrib = actualTraces.pn_distribution(normalised=True)
        yvals = np.zeros_like(cal_distrib)
        yvals[:len(predicted_distrib)] = predicted_distrib
        print(f'Predicted pn distribution = {yvals}')

        '''Save results'''
        results_df.loc[i_rep] = [high_rep_rate, actualTraces.num_traces, accuracy, t2-t1, t4-t3] + list(yvals)
        results_df.to_csv(DFUtils.create_filename(results_dir + rf'\{modeltype}_results_{data_group}.csv'), index=False)

        mlClassifier.save(filename=rf'{modeltype}_trained_by_{data_group}_{high_rep_rate}kHz', filedir=results_dir + r'\saved_classifiers')

