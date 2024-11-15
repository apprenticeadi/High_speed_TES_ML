import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils import poisson_norm, tvd

'''Script to compare all models by TVD and processing time'''

ref_model = 'IP'
models = ['IP', 'RF', 'BDT', 'SVM', 'KNN', 'CNN']  # , 'HDBSCAN']
hdbscan_rep_rates = [100, 500, 800]

power = 10 #  np.arange(10)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

rep_rates = np.arange(100, 1100, 100)

params_dir = rf'..\..\Results\Tomography_data_2024_04\Params'
log_df = pd.read_csv(params_dir + r'\log_2024_04_22.csv')
save_df = rf'../../Plots/Tomography_data_2024_04/tvd_all_models'
os.makedirs(save_df, exist_ok=True)
# compare_with_pm = False

# results
tvd_df = pd.DataFrame(columns=['rep_rate', 'pm_mu', 'ip_mu'] + models)
tvd_df['rep_rate'] = rep_rates
for rep_rate in rep_rates:
    pm_mu = log_df.loc[(log_df['power_group'] == f'power_{power}') & (log_df['rep_rate/kHz'] == rep_rate), 'pm_estimated_av_pn'].iloc[0]
    tvd_df.loc[tvd_df['rep_rate'] == rep_rate, 'pm_mu'] = pm_mu

training_t_df = pd.DataFrame(columns=['rep_rate'] + models[:-1])  # hdbscan processing time calculated differently
training_t_df['rep_rate'] = rep_rates[1:]  # exclude 100 kHz
predict_t_df = pd.DataFrame(columns=['rep_rate'] + models[:-1])  # hdbscan processing time calculated differently
predict_t_df['rep_rate'] = rep_rates[1:]  # exclude 100 kHz

fontsize = 14
fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')

ref_mu = np.nan
df_ref = pd.read_csv(params_dir + rf'\{ref_model}\{ref_model}_results_power_{power}.csv')
ip_distrib_100 = np.array(df_ref.loc[df_ref['rep_rate'] == 100, '0':].iloc[0])  # ip distribution at 100 kHz
ip_distrib_100 = np.nan_to_num(ip_distrib_100)

for i_model, ml_model in enumerate(models):
    # ML data
    df_ml = pd.read_csv(params_dir + rf'\{ml_model}\{ml_model}_results_power_{power}.csv')
    # Calculate TVD for each rep rate
    for i_rep, rep_rate in enumerate(rep_rates):
        if ml_model == 'HDBSCAN' and rep_rate not in hdbscan_rep_rates:
            tvd_i = np.nan
        else:
            # get ML distribution
            ml_distrib = np.array(df_ml.loc[df_ml['rep_rate'] == int(rep_rate), '0':].iloc[0])
            ml_distrib = np.nan_to_num(ml_distrib)
            ml_labels = np.arange(len(ml_distrib))

            if ml_model == 'IP':
                ip_mu = np.sum(ml_distrib * ml_labels)
                tvd_df.loc[tvd_df['rep_rate'] == rep_rate, 'ip_mu'] = ip_mu
                if rep_rate == 100:
                    ref_mu = ip_mu

            if rep_rate != 100 and ml_model != 'HDBSCAN':
                training_t_df.loc[training_t_df['rep_rate'] == rep_rate, ml_model] = \
                    df_ml.loc[df_ml['rep_rate'] == int(rep_rate), 'training_t'].iloc[0] / df_ml.loc[df_ml['rep_rate'] == int(rep_rate), 'num_traces'].iloc[0]
                predict_t_df.loc[predict_t_df['rep_rate'] == rep_rate, ml_model] = \
                    df_ml.loc[df_ml['rep_rate'] == int(rep_rate), 'predict_t'].iloc[0] / df_ml.loc[df_ml['rep_rate'] == int(rep_rate), 'num_traces'].iloc[0]

            # calculate tvd
            # tvd_i = tvd(ml_distrib, poisson_norm(ml_labels, ref_mu))
            tvd_i = tvd(ml_distrib, ip_distrib_100)

        tvd_df.loc[tvd_df['rep_rate'] == rep_rate, ml_model] = tvd_i

    # Plot TVD
    if ml_model == 'HDBSCAN':
        ax.plot(hdbscan_rep_rates, tvd_df.loc[tvd_df['rep_rate'].isin(hdbscan_rep_rates), ml_model], 'o:', alpha=0.5, label=f'{ml_model}')
    elif ml_model == 'IP':
        ax.plot(rep_rates, tvd_df[ml_model], 'o--', alpha=0.5, markersize=6, label=f'{ml_model}')
    else:
        ax.plot(rep_rates, tvd_df[ml_model], 'o-', alpha=0.8, markersize=6, label=f'{ml_model}')

ax.legend(loc='upper left', fontsize=fontsize - 2)
ax.set_xlabel('Repetition rate/kHz', fontsize=fontsize)
ax.set_ylabel('TVD', fontsize=fontsize)
ax.set_xticks(rep_rates)
ax.tick_params(labelsize=fontsize - 2)
ax.set_title(r'$\mu=$' + f'{ref_mu:.3g}', fontsize=fontsize+2)
plt.show()
#
# fig2, ax2 = plt.subplots(figsize=(8, 4), layout='constrained')
# ax2.bar(np.arange(len(models)), processing_ts, width=0.5, align='center')
# for i in range(len(models)):
#     ax2.text(i, processing_ts[i], f'{processing_ts[i]:.2f}', ha='center', va='bottom', fontsize=fontsize - 2)
# ax2.set_xticks(np.arange(len(models)))
# ax2.set_xticklabels(models)
# ax2.set_ylabel('Processing time/s', fontsize=fontsize)
# ax2.set_title('Average processing time for 100,000 traces', fontsize=fontsize+2)
# ax2.tick_params(labelsize=fontsize-2)

# tvd_df.to_csv(save_df + rf'\tvd_power_{power}.csv', index=False)
# training_t_df.to_csv(save_df + rf'\training_t_per_trace_power_{power}.csv', index=False)
# predict_t_df.to_csv(save_df + rf'\predict_t_per_trace_power_{power}.csv', index=False)
# fig.savefig(save_df + rf'\tvd_power_{power}.png')

