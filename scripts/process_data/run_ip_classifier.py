import numpy as np
import matplotlib.pyplot as plt
import time
import os

import pandas as pd

from tes_resolver import Traces, DataChopper, config
from tes_resolver.classifier import InnerProductClassifier

from utils import DFUtils, DataReader

'''Run inner product classifier to classify all the data in a certain folder. '''
# data parameters
modeltype = 'IP'

sampling_rate = 5e4
data_name = 'Tomography_data_2024_04'
dataReader = DataReader(f'Data/{data_name}')
powers = np.arange(0, 12)
data_groups = np.array([f'power_{p}' for p in powers])  # different groups of coherent states

rep_rates = np.arange(100, 1100, 100)  # the higher rep rates to predict
mosaic = rep_rates.reshape(2,5)

update_params = False  # whether to store the results in a params folder
if update_params:
    params_dir = os.path.join(config.home_dir, '..', 'Results', data_name, 'Params', modeltype)

for data_group in data_groups:
    print(f'\nProcessing {data_group}...')
    # save data
    results_dir = os.path.join(config.home_dir, '..', 'Results', data_name, modeltype, f'{data_group}_{config.time_stamp}')

    # Result file
    results_df = pd.DataFrame(columns=['rep_rate', 'num_traces', 'acc_score', 'training_t', 'predict_t'])

    # Plotting
    fig, axs = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, figsize=(20, 8), layout='constrained')

    for i_rep, rep_rate in enumerate(rep_rates):

        # Load actual traces
        ti = time.time()
        actual_data = dataReader.read_raw_data(data_group, rep_rate)

        # set suitable trigger delay
        if rep_rate <= 300:
            trigger_delay = 0
        else:
            trigger_delay = DataChopper.find_trigger(actual_data, samples_per_trace=int(sampling_rate/rep_rate))

        actualTraces = Traces(rep_rate, actual_data, parse_data=True, trigger_delay=trigger_delay)
        tf = time.time()
        print(f'\nLoad high rep rate data into traces took {tf-ti}s')

        # Run inner product classifier
        t1 = time.time()
        ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)

        # Train
        print(f'Training classifier for {rep_rate}kHz')
        ipClassifier.train(actualTraces)
        t2 = time.time()

        # Predict
        print(f'Making predictions for {actualTraces.num_traces} traces')
        t3 = time.time()
        raw_labels = ipClassifier.predict(actualTraces, update=True)
        t4 = time.time()
        print(f'Training finished after {t2-t1}s. Prediction finished after {t4-t3}s. ')

        # results
        pn_labels, predicted_distrib = actualTraces.pn_distribution(normalised=True)
        print(f'Predicted pn distribution = {predicted_distrib}')

        # save results
        results_df.loc[i_rep, :'predict_t'] = [rep_rate, actualTraces.num_traces, np.nan, t2-t1, t4-t3]
        for i_label, label in enumerate(pn_labels):
            if label in results_df.columns:
                results_df.loc[i_rep, label] = predicted_distrib[i_label]
            else:
                results_df[label] = i_rep * [0.] + [predicted_distrib[i_label]]

        results_df.to_csv(DFUtils.create_filename(results_dir + rf'\{modeltype}_results_{data_group}.csv'), index=False)
        np.save(results_dir + rf'\{modeltype}_{data_group}_{rep_rate}kHz_raw_labels.npy', raw_labels)

        # Plot stegosaurus
        ax = axs[rep_rate]
        overlaps = ipClassifier.calc_inner_prod(actualTraces)
        ax.hist(overlaps, bins=ipClassifier.num_bins, color='aquamarine')
        for pn_bin in ipClassifier.inner_prod_bins.values():
            ax.axvline(pn_bin, ymin=0., ymax=0.5, ls='dashed', color='black')
        ax.set_xlabel('Inner product')
        ax.set_ylabel('Counts')
        ax.set_title(f'{rep_rate}kHz', loc='left')

    fig.savefig(results_dir + rf'\{modeltype}_stegosauruses.pdf')

    if update_params:
        results_df.to_csv(DFUtils.create_filename(params_dir + rf'\{modeltype}_results_{data_group}.csv'), index=False)
