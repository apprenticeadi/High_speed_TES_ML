import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import factorial

from src.utils import DFUtils, poisson_norm, tvd

ref_model = 'IP'
models = ['IP', 'RF', 'BDT', 'SVM', 'CNN']

power =  9 #  np.arange(10)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

rep_rates = np.arange(100, 1100, 100)

params_dir = rf'..\..\Results\Tomography_data_2024_04\Params'
# log_df = pd.read_csv(params_dir + r'\log_2024_04_22.csv')
# compare_with_pm = False


fontsize = 14
fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')

ref_mu = np.nan
df_ref = pd.read_csv(params_dir + rf'\{ref_model}\{ref_model}_results_power_{power}.csv')
processing_ts = np.zeros(len(models))
for i_model, ml_model in enumerate(models):

    # ML data
    df_ml = pd.read_csv(params_dir + rf'\{ml_model}\{ml_model}_results_power_{power}.csv')
    tvds = np.zeros(len(rep_rates))  # first row ref, second row ml
    # Calculate TVD for each rep rate
    for i_rep, rep_rate in enumerate(rep_rates):
        # get reference distribution
        ref_distrib = np.array(df_ref.loc[df_ref['rep_rate'] == int(rep_rate), '0':].iloc[0])
        ref_distrib = np.nan_to_num(ref_distrib)  # nan to 0s.
        ref_labels = np.arange(len(ref_distrib))

        if rep_rate == 100:
            ref_mu = np.sum(ref_distrib * ref_labels)

        # get ML distribution
        ml_distrib = np.array(df_ml.loc[df_ml['rep_rate'] == int(rep_rate), '0':].iloc[0])
        ml_distrib = np.nan_to_num(ml_distrib)
        ml_labels = np.arange(len(ml_distrib))

        if rep_rate != 100:
            processing_ts[i_model] += df_ml.loc[df_ml['rep_rate'] == int(rep_rate), 'training_t'].iloc[0] \
                           + df_ml.loc[df_ml['rep_rate'] == int(rep_rate), 'predict_t'].iloc[0]

        # calculate tvd
        tvds[i_rep] = tvd(ml_distrib, poisson_norm(ml_labels, ref_mu))


    # Plot TVD
    ax.plot(rep_rates, tvds, 'o-',  alpha=0.8, markersize=6, label=f'{ml_model}')

ax.legend(loc='upper left', fontsize=fontsize - 2)
ax.set_xlabel('Repetition rate/kHz', fontsize=fontsize)
ax.set_ylabel('TVD', fontsize=fontsize)
ax.set_xticks(rep_rates)
ax.tick_params(labelsize=fontsize - 2)
ax.set_title(r'$\mu=$' + f'{ref_mu:.3g}', fontsize=fontsize+2)
plt.show()

processing_ts = processing_ts / len(rep_rates)
fig2, ax2 = plt.subplots(figsize=(8, 4), layout='constrained')
ax2.bar(np.arange(len(models)), processing_ts, width=0.5, align='center')
ax2.set_xticks(np.arange(len(models)))
ax2.set_xticklabels(models)
ax2.set_ylabel('Processing time/s', fontsize=fontsize)
ax2.set_title('Average processing time for 100,000 traces', fontsize=fontsize+2)
ax2.tick_params(labelsize=fontsize-2)
