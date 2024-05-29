from tes_resolver.traces import Traces
from tes_resolver.classifier import InnerProductClassifier
from tes_resolver.data_chopper import DataChopper
from src.data_reader import DataReader
from src.utils import DFUtils
import tes_resolver.config as config

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

powers = np.arange(12)
rep_rates = np.arange(100, 1100, 100)
modeltype = 'RF'

bootstrap_for_mean = False

params_dir = rf'..\..\Results\Tomography_data_2024_04\Params\{modeltype}'

for power in powers:
    results_df = pd.read_csv(params_dir + fr'\{modeltype}_results_power_{power}.csv')
    pn_labels = np.arange(results_df.loc[:, '0':].shape[1])

    if bootstrap_for_mean:
        means_df = pd.DataFrame(columns=['rep_rate', 'mean_pn', 'n_error', 'p_error'])
    else:
        n_errors_df = pd.DataFrame(columns= ['rep_rate'] + list(pn_labels))
        p_errors_df = pd.DataFrame(columns= ['rep_rate'] + list(pn_labels))

    for i_reprate, rep_rate in enumerate(rep_rates):
        num_traces = results_df.loc[results_df['rep_rate']==rep_rate, 'num_traces'].iloc[0]
        num_traces = int(num_traces)
        pn_distrib = np.array(results_df.loc[results_df['rep_rate'] == rep_rate, '0':].iloc[0])
        pn_distrib = np.nan_to_num(pn_distrib)

        mean_pn = np.sum(pn_labels * pn_distrib)
        pn_counts = np.rint(pn_distrib * num_traces).astype(int)

        raw_labels_mimic = np.zeros(num_traces, dtype=int)
        start_id = 0
        for pn, pn_count in enumerate(pn_counts):
            raw_labels_mimic[start_id: start_id + pn_count] = [pn] * pn_count
            start_id = start_id + pn_count

        np.random.shuffle(raw_labels_mimic)

        if bootstrap_for_mean:
            print(f'Bootstrapping for mean_pn at {rep_rate}kHz, power_{power}')
            res = bootstrap((raw_labels_mimic,), np.mean, confidence_level=0.95, batch=100, method='basic')
            means_df.loc[i_reprate] = [rep_rate, mean_pn, mean_pn - res.confidence_interval.low, res.confidence_interval.high - mean_pn]

            means_df.to_csv(DFUtils.create_filename(params_dir + rf'\bootstrapped_means\{modeltype}_results_power_{power}.csv'))

        else:
            n_errors = np.zeros(len(pn_labels))
            p_errors = np.zeros(len(pn_labels))
            for pn in pn_labels:
                if pn_distrib[pn] == 0:
                    pass
                else:
                    data = raw_labels_mimic == pn
                    print(f'Bootstrapping for pn={pn} at {rep_rate}kHz, power_{power}')
                    res = bootstrap((data,), np.mean, confidence_level=0.95, batch=100, method='basic')
                    n_errors[pn] = pn_distrib[pn] - res.confidence_interval.low
                    p_errors[pn] = res.confidence_interval.high - pn_distrib[pn]

            n_errors_df.loc[i_reprate] = [rep_rate] + list(n_errors)
            p_errors_df.loc[i_reprate] = [rep_rate] + list(p_errors)

            n_errors_df.to_csv(DFUtils.create_filename(params_dir + rf'\bootstrapped\{modeltype}_results_power_{power}_n_error.csv'))
            p_errors_df.to_csv(params_dir + rf'\bootstrapped\{modeltype}_results_power_{power}_p_error.csv')


