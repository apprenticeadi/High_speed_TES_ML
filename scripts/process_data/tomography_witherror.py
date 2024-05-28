import time

import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import logging
from scipy.special import factorial

from tomography import params_dir, log_df, tomography, construct_guess_theta, fidelity_by_n
from src.utils import DFUtils, LogUtils


modeltype = 'IP'
powers = np.arange(11)
rep_vals = np.arange(100, 900, 100)
repeat = 10

guess_efficiency = 0.93
pm_error = 0.046

time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
results_dir = params_dir + rf'\..\tomography\witherror\tomography_on_{modeltype}_{time_stamp}'

'''Define truncation'''
max_input = 8
max_detected = 8
assert max_input >= max_detected

indices = [f'{i}' for i in range(max_detected + 1)] + [f'{max_detected + 1}+']
columns = [f'{i}' for i in range(max_input + 1)] + [f'{max_input + 1}+']

''' Define guess values '''
guess_theta = construct_guess_theta(guess_efficiency, max_detected, max_input)
guess_df = pd.DataFrame(data=guess_theta, index=indices, columns=columns)
guess_df.to_csv(DFUtils.create_filename(results_dir + rf'\guess_theta.csv'))

'''Logging'''
LogUtils.log_config(time_stamp='', dir=results_dir, filehead='log', module_name='', level=logging.INFO)
logging.info(
    f'Run tomography with error bars on data from power_{powers}, rep rates = {rep_vals}, processed with {modeltype} classifier. \n'
    f'Input state characterised by power meter with Gaussian error={pm_error * 100}%. \n'
    f'Input state are sampled from Gaussian distribution on the power meter error. Detected probabilities are errorbarred by bootstrapping. \n'
    f'Tomography is repeated for {repeat} times to give errorbars. \n'
    f'For tomography, input photon number truncated at max_input={max_input}, detected photon number truncated at max_detected={max_detected}. ')

fig = plt.figure(figsize=(16, 8))
final_costs = np.zeros((len(rep_vals), repeat))
for i_rep, rep_rate in enumerate(rep_vals):

    '''Read in input state and detected probabilities'''
    mean_pns = np.zeros(len(powers))
    probs = np.zeros((max_detected + 2, len(powers)))  # last row is for all photons beyond max_detected+1.
    probs_error = np.zeros_like(probs)
    for i_power, power in enumerate(powers):
        mean_pns[i_power] = log_df.loc[(log_df['power_group'] == f'power_{power}') & (log_df['rep_rate/kHz'] == rep_rate), 'pm_estimated_av_pn'].iloc[0]

        probs_df = pd.read_csv(params_dir + rf'\{modeltype}\{modeltype}_results_power_{power}.csv')
        distrib = np.array(probs_df.loc[probs_df['rep_rate'] == rep_rate, '0':].iloc[0])  # the probability distribution
        distrib = np.nan_to_num(distrib)

        distrib_errors = np.zeros((2, len(distrib)))
        for i_n_or_p, n_or_p in enumerate(['n', 'p']):
            error_df = pd.read_csv(params_dir + rf'\{modeltype}\bootstrapped\{modeltype}_results_power_{power}_{n_or_p}_error.csv')
            distrib_errors[i_n_or_p] = np.array(error_df.loc[error_df['rep_rate'] == rep_rate, '0':].iloc[0])
        distrib_errors = np.nan_to_num(distrib_errors)
        distrib_errors = np.max(distrib_errors, axis=0)

        if len(distrib) <= max_detected + 2:
            probs[:len(distrib), i_power] = distrib
            probs_error[:len(distrib), i_power] = distrib_errors
        else:
            probs[:-1, i_power] = distrib[:max_detected + 1]
            probs_error[:-1, i_power] = distrib_errors[:max_detected + 1]

            probs[-1, i_power] = np.sum(distrib[max_detected + 1:])
            probs_error[-1, i_power] = np.sqrt(np.sum(np.square(distrib_errors[max_detected + 1:])))

    probs_df = pd.DataFrame(data=probs, index=indices, columns=[f'power_{p_ind}' for p_ind in powers])
    probs_df.to_csv(DFUtils.create_filename(results_dir + rf'\{rep_rate}kHz_probs.csv'))

    probs_error_df = pd.DataFrame(data=probs_error, index=indices, columns=[f'power_{p_ind}' for p_ind in powers])
    probs_error_df.to_csv(results_dir + rf'\{rep_rate}kHz_probs_error.csv')

    '''Repeat tomography routine for multiple times with different errors'''
    estimated_thetas = np.zeros((repeat, guess_theta.shape[0], guess_theta.shape[1]))
    optimal_costs = np.zeros(repeat)
    for i_repeat in range(repeat):

        '''Gaussian error on input state'''
        er_mean_pns = np.random.normal(mean_pns, pm_error * mean_pns)

        '''Gaussian error on detected probabilities '''
        er_probs = np.random.normal(probs, probs_error)
        er_probs = er_probs / np.sum(er_probs, axis=0)  # renormalise

        '''Calculate F matrix'''
        F_matrix = np.zeros((max_input + 2, len(powers)))
        ms = np.arange(max_input + 1)
        for i_power in range(len(powers)):
            F_matrix[:-1, i_power] = np.exp(- er_mean_pns[i_power]) * np.power(er_mean_pns[i_power], ms) / factorial(ms)
            F_matrix[-1, i_power] = 1 - np.sum(F_matrix[:-1, i_power])

        '''Tomography'''
        t1 = time.time()
        estimated_thetas[i_repeat], optimal_costs[i_repeat] = tomography(er_probs, F_matrix, guess_theta)
        t2 = time.time()
        logging.info(f'{i_repeat} repeat for {rep_rate}kHz, tomography finished after {t2-t1}s with optimal_cost={optimal_costs[i_repeat]}')

    final_costs[i_rep] = optimal_costs

    # save estimated thetas
    np.save(results_dir + rf'\{rep_rate}kHz_estimated_thetas.npy', estimated_thetas)

    theta_mean = np.mean(estimated_thetas, axis=0)
    theta_std = np.std(estimated_thetas, axis=0)

    ax = fig.add_subplot(2, 4, i_rep+1, projection='3d')

    _x = np.arange(theta_mean.shape[1])
    _y = np.arange(theta_mean.shape[0])
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    ax.bar3d(x-0.5, y, np.zeros_like(theta_mean).ravel(), 1, 0, theta_mean.ravel(), shade=True, color='cornflowerblue', alpha=0.6)

    for i, j in zip(x,y):
        ax.plot([i,i], [j,j], [theta_mean[i,j] - theta_std[i,j], theta_mean[i,j] + theta_std[i,j]], marker='_', color='red')

    ax.set_zlim(0,1)
    ax.set_xticks(_x)
    ax.set_xticklabels(columns)
    ax.set_xlabel('Input photon')

    ax.set_yticks(_y)
    ax.set_yticklabels(indices)
    ax.set_ylabel('Detected photon')

fig.savefig(results_dir + rf'\theta_3d_with_errors.pdf')

np.save(results_dir + rf'\final_costs.npy', final_costs)

plt.show()




# TODO: sample a probs from bootstrapping result, and an F from Gaussian distribution of mean photon number (error on attenuation from repeatibility and linearity of PM, error on PM reading from calibration error). Run tomography many times and save them. Plot 3D bar plot with error bars.

