import time
import numpy as np
import datetime
import pandas as pd
import logging
from scipy.special import factorial
import string

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from tomography import params_dir, log_df, tomography, construct_guess_theta, fidelity_by_n
from src.utils import DFUtils, LogUtils

modeltype = 'RF'
powers = np.arange(11)
rep_vals = np.arange(100, 900, 100)
repeat = 100

guess_efficiency = 0.93
pm_error = 0.046

time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
results_dir = params_dir + rf'\..\tomography\witherror\tomography_on_{modeltype}_{time_stamp}'

'''Define truncation'''
max_input = 16
max_detected = 16
assert max_input >= max_detected

indices = [f'{i}' for i in range(max_detected + 1)] + [f'{max_detected + 1}+']
columns = [f'{i}' for i in range(max_input + 1)] + [f'{max_input + 1}+']

''' Define guess values '''
guess_theta = construct_guess_theta(guess_efficiency, max_detected, max_input)
guess_df = pd.DataFrame(data=guess_theta, index=indices, columns=columns)
guess_df.to_csv(DFUtils.create_filename(results_dir + rf'\guess_theta.csv'))

ideal_theta = construct_guess_theta(1., max_detected, max_input)
ref_theta = guess_theta  # for relative fidelity calculation

'''Logging'''
LogUtils.log_config(time_stamp='', dir=results_dir, filehead='log', module_name='', level=logging.INFO)
logging.info(
    f'Run tomography with error bars on data from power_{powers}, rep rates = {rep_vals}, processed with {modeltype} classifier. \n'
    f'Input state characterised by power meter with Gaussian error={pm_error * 100}%. \n'
    f'Input state are sampled from Gaussian distribution on the power meter error. Detected probabilities are errorbarred by bootstrapping. \n'
    f'Tomography is repeated for {repeat} times to give errorbars. \n'
    f'For tomography, input photon number truncated at max_input={max_input}, detected photon number truncated at max_detected={max_detected}. ')

'''Figure'''
fig = plt.figure(figsize=(18, 8))
width = 0.4  # 3d bar plot width
depth = 0.2  # 3d bar plot depth

cmap = plt.get_cmap()
norm = Normalize(vmin=0., vmax=1.)
alphabet = list(string.ascii_lowercase)

'''Iterate over rep rates'''
final_costs = np.zeros((len(rep_vals), repeat))
fidelities = np.zeros((len(rep_vals), repeat, max_detected + 2))  # fidelity with ideal POVM
rel_fidelities = np.zeros((len(rep_vals)-1, repeat, max_detected + 2))  # fidelity with average 100kHz POVM
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
    t1 = time.time()
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
        theta_rec, optimal_costs[i_repeat] = tomography(er_probs, F_matrix, guess_theta)
        estimated_thetas[i_repeat] = theta_rec

        '''Calculate fidelity'''
        fidelities[i_rep, i_repeat, :] = fidelity_by_n(theta_rec, ideal_theta)
        if rep_rate > 100:
            rel_fidelities[i_rep-1, i_repeat, :] = fidelity_by_n(theta_rec, ref_theta)

    t2 = time.time()
    logging.info(f'{repeat} repeats for {rep_rate}kHz, tomography finished after {t2-t1}s with average optimal_cost={np.mean(optimal_costs)}')

    # final cost
    final_costs[i_rep] = optimal_costs

    # relative fidelity
    if rep_rate == 100:
        ref_theta = np.mean(estimated_thetas, axis=0)

    # save estimated thetas
    np.save(results_dir + rf'\{rep_rate}kHz_estimated_thetas.npy', estimated_thetas)

    theta_mean = np.mean(estimated_thetas, axis=0)
    theta_std = np.std(estimated_thetas, axis=0)

    '''Plot POVM'''
    ax = fig.add_subplot(2,4, i_rep+1, projection='3d')

    _x = np.arange(theta_mean.shape[1])
    _y = np.arange(theta_mean.shape[0])
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    colors = cmap(norm(theta_mean.ravel()))
    ax.bar3d(x - width/2, y - depth/2, np.zeros_like(theta_mean).ravel(), width, depth, theta_mean.ravel(), shade=True, color=colors, alpha=0.8)

    for i, j in zip(x,y):
        low = np.max([theta_mean[j,i] - theta_std[j,i], 0.])
        high = np.min([1., theta_mean[j,i] + theta_std[j,i]])
        if high-low > 0.1:
            ax.plot([i,i], [j,j], [low, high], marker='_', color='red')

    ax.set_zlim(0,1)
    ax.set_xticks(_x)
    ax.set_xticklabels(columns)
    ax.set_xlabel('Input photon')
    ax.tick_params(axis='x', which='major', pad=-3)
    ax.xaxis.labelpad = -5

    ax.set_yticks(_y)
    ax.set_yticklabels(indices, verticalalignment='baseline', horizontalalignment='left')
    ax.set_ylabel('Detected photon')

    ax.set_title(f'({alphabet[i_rep]}) {rep_rate}kHz')

fig.savefig(results_dir + rf'\theta_3d_with_errors.pdf')

np.save(results_dir + rf'\final_costs.npy', final_costs)
np.save(results_dir + rf'\abs_fidelities_by_n.npy', fidelities)  # fidelity with ideal POVM
np.save(results_dir + rf'\rel_fidelities_by_n.npy', rel_fidelities)  # relative fidelity with 100kHz


'''Plot absolute fidelity'''
av_fidelities = np.mean(fidelities, axis=1)[::-1]  # average over the repeats.
std_fidelities = np.std(fidelities, axis=1)[::-1]

fig2 = plt.figure(figsize=(6,6))
ax2 = fig2.add_subplot(111, projection='3d')

_x2 = np.arange(av_fidelities.shape[1])  # number of detected photons
_y2 = np.arange(av_fidelities.shape[0])  # number of rep rates
_xx2, _yy2 = np.meshgrid(_x2, _y2)
x2, y2 = _xx2.ravel(), _yy2.ravel()

ax2.bar3d(x2 - width/2, y2 - depth/2, np.zeros_like(av_fidelities).ravel(), width, depth, av_fidelities.ravel(), shade=True, color=cmap(norm(av_fidelities.ravel())), alpha=0.8)
for i, j in zip(x2, y2):
    low = np.max([av_fidelities[j, i] - std_fidelities[j, i], 0.])
    high = np.min([1., av_fidelities[j, i] + std_fidelities[j, i]])
    ax2.plot([i, i], [j, j], [low, high], marker='_', color='red')

ax2.set_zlim(0,1)
ax2.set_xticks(_x2)
ax2.set_xticklabels(columns)
ax2.set_xlabel('Detected photon')

ax2.set_yticks(_y2)
ax2.set_yticklabels(rep_vals[::-1], verticalalignment='baseline', horizontalalignment='left')
ax2.set_ylabel('Rep rate (kHz)')

fig2.savefig(results_dir + rf'\absolute_fidelity_by_n.pdf')

'''Plot relative fidelity'''
fidelity_calculation_cutoff_n = max_detected
av_rel_fidelities = np.mean(rel_fidelities[:, :, :fidelity_calculation_cutoff_n+1], axis=2)  # average over n

av_rel_fidelity = np.mean(av_rel_fidelities, axis=1)  # average over repeats
std_rel_fidelity = np.std(av_rel_fidelities, axis=1)

fig3, ax3 = plt.subplots()
ax3.errorbar(rep_vals[1:], av_rel_fidelity, yerr=std_rel_fidelity, fmt='.', ls='None')
ax3.set_xticks(rep_vals[1:])
ax3.set_xlabel('Rep rate (kHz)')
ax3.set_ylabel('Fidelity')
ax3.set_title('Relative fidelity')
ax3.set_ylim(0,1)
fig3.savefig(results_dir + rf'\relative_fidelity.pdf')

plt.show()

