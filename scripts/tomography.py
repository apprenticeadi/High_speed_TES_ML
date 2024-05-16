from scipy.special import factorial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import comb
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cvxpy as cp
import logging
import time
import datetime
import string

from src.utils import LogUtils, DFUtils

#TODO: for higher rep rates, the convergence by cvxpy is just bad, but this makes the final thetas look ok. How to account for this? Read Humphreys again.

'''
script to perform a tomography routine on the probabilities, using the cvxpy package
'''
modeltype = 'RF'

alphabet = list(string.ascii_lowercase)

powers = np.arange(11)
rep_vals = [100, 500, 800] #  np.arange(100, 900, 100)

time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
results_dir = rf'..\Results\Tomography_data_2024_04\tomography_{time_stamp}'

params_dir = rf'..\Results\Tomography_data_2024_04\Params'
log_df = pd.read_csv(params_dir + r'\log_2024_04_22.csv')

'''Define truncation'''
max_input = 16 # truncated input number, will keep a max_input+1+
max_detected = 16 # truncated detected number, will keep a max_detected+1+
assert max_input >= max_detected

indices = [f'{i}' for i in range(max_detected + 1)] + [f'{max_detected + 1}+']
columns = [f'{i}' for i in range(max_input + 1)] + [f'{max_input + 1}+']

'''
Theta is the POVM elements that we wish to find, dimension N+2 * M+2, N=max_detect, M=max_input
Theta_nm is the probability that the detector measures n photon, given m photon input. 
sum over column should be normalised to 1
Define guess values
'''
guess_efficiency = 0.95
guess_theta = np.zeros((max_detected + 2, max_input + 2))
for i in range(guess_theta.shape[0]):
    if i < max_detected + 1:
        guess_theta[i, i:] = [comb(j, i) * ((1-guess_efficiency)**(j-i)) * (guess_efficiency ** i) for j in range(i, max_input+2)]
    else: # last row
        guess_theta[i, i:] = 1. - np.sum(guess_theta[:i, i:], axis=0)
guess_df = pd.DataFrame(data=guess_theta, index=indices, columns=columns)
guess_df.to_csv(DFUtils.create_filename(results_dir + rf'\guess_theta.csv'))


'''Logging'''
LogUtils.log_config(time_stamp='', dir=results_dir, filehead='log', module_name='', level=logging.INFO)
logging.info(f'Run tomography on data from power_{powers}, rep rates = {rep_vals}, processed with {modeltype} classifier. For tomography, '
             f'input photon number truncated at max_input={max_input}, detected photon number truncated at max_detected={max_detected}. '
             # f'Input light characterised by 100kHz inner product method instead of power meter. '
             )

# fig, axs = plt.subplots(2, 4, squeeze=True, sharey=True, sharex=True, layout='constrained', figsize=(15,7))
fig, axs = plt.subplots(1, 3, squeeze=True, sharey=True, sharex=True, layout='constrained', figsize=(12, 4))
axs = axs.flatten()

final_costs = np.zeros(len(rep_vals))
fidelities = np.zeros((max_detected+2, len(rep_vals)))  # fidelity with respect to ideal (delta function)
rel_fidelities = np.zeros((max_detected+2, len(rep_vals)))  # fidelity with respect to 100kHz for each POVM_n, where n is the reported photon number
theta_ref = np.zeros_like(guess_theta)
for i_rep, rep_rate in enumerate(rep_vals):

    '''
    Load experimental probabilites- matrix P, with dimensions (N+2)*S, where N is the max_detected, S is the 
    number of powers. 
    P_ns is the experimental probability of measuring n photons from coherent light with power s=|alpha|^2
    '''
    probs = np.zeros((len(powers), max_detected + 2))  # later transposed, last column is for all photons beyond max_detected+1.
    mean_pns = np.zeros(len(powers))
    for i_power, power in enumerate(powers):
        # mean_pns[i_power] = log_df.loc[(log_df['power_group'] == f'power_{power}') & (log_df['rep_rate/kHz'] == rep_rate), 'pm_estimated_av_pn'].iloc[0]
        mean_pns[i_power] = log_df.loc[(log_df['power_group'] == f'power_{power}') & (log_df['rep_rate/kHz'] == 100), 'ip_classifier_av_pn'].iloc[0]

        df = pd.read_csv(params_dir + rf'\{modeltype}\{modeltype}_results_power_{power}.csv')
        pn = np.array(df.loc[df['rep_rate'] == rep_rate, '0':].iloc[0])  # the resolved photon number distribution
        if len(pn) <= max_input + 2:
            probs[i_power, :len(pn)] = pn
        else:
            probs[i_power, :-1] = pn[:max_input+1]
            probs[i_power, -1] = np.sum(pn[max_input+1:])

    probs = probs.T

    '''
    Calculate F matrix, dimension (M+2)*S, where M is the max_input. 
    F_ms is the theoretical (Poissonian) probability of inputting m photons from coherent light with power s=|alpha|^2. 
    F is normalised, so the last row is 1-sum of all previous rows, and represents the probability of inputting max_input+1 and beyond photons. 
    '''
    F = np.zeros((len(powers), max_input + 2))
    ms = np.arange(max_input + 1)
    for i_power in range(len(powers)):
        F[i_power, :-1] = np.exp(- mean_pns[i_power]) * np.power(mean_pns[i_power], ms) / factorial(ms)
        F[i_power, -1] = 1 - np.sum(F[i_power, :-1])
    F = F.T

    '''
    cvxpy least squares minimization, 
    cost function as a sum of squares
    constraints that sum over rows and columns equal 1
    '''
    theta = cp.Variable(guess_theta.shape, nonneg=True, value=guess_theta)
    cost = cp.sum_squares(cp.abs(probs - theta @ F))
    constraints = [0 <= theta, theta <= 1, cp.sum(theta, axis=0) == 1]
    problem = cp.Problem(cp.Minimize(cost), constraints)

    t1 = time.time()
    optimal_value = problem.solve()
    t2 = time.time()
    msg = f'For {rep_rate}kHz, min squares routine finish after {t2 - t1}s, optimal_value = {optimal_value}'
    logging.info(msg)
    final_costs[i_rep] = optimal_value

    estimated_theta = theta.value
    if i_rep == 0:
        theta_ref = estimated_theta

    # np.save(DFUtils.create_filename(results_dir + rf'\{rep_rate}kHz_theta.npy'), estimated_theta)
    theta_df = pd.DataFrame(data=estimated_theta, index=indices, columns=columns)
    theta_df.to_csv(results_dir + rf'\{rep_rate}kHz_theta.csv')

    '''Calculate fidelity'''
    for i in range(max_detected+2):
        fidelities[i, i_rep] = estimated_theta[i,i] / np.sum(estimated_theta[i, :])
        rel_fidelities[i, i_rep] = np.sum(np.sqrt(theta_ref[i, :] * estimated_theta[i, :]))**2 / (np.sum(theta_ref[i, :]) * np.sum(estimated_theta[i, :]))

    '''
    create colour plot, using same blue to yellow colours as White paper
    '''
    # fig, ax = plt.subplots()
    ax = axs[i_rep]

    x = np.arange(estimated_theta.shape[1])
    y = np.arange(estimated_theta.shape[0])
    X, Y = np.meshgrid(x, y)

    pc = ax.pcolormesh(X, Y, estimated_theta,
                       # norm=mcolors.SymLogNorm(linthresh=0.01),
                       vmin=0., vmax=1.,
                       )

    ax.set_xticks(x[1::2])
    ax.set_xticklabels(columns[1::2])
    ax.set_yticks(y)
    ax.set_yticklabels(indices)

    # ax.set_title(rf'({alphabet[i_rep]}) {rep_rate}kHz, fidelity={fidelity:.3g}, final cost={optimal_value:.2g}')
    if rep_rate==100:
        ax.set_title(rf'({alphabet[i_rep]}) {rep_rate}kHz', loc='left')
    else:
        ax.set_title(rf'({alphabet[i_rep]}) {rep_rate}kHz, fidelity={np.mean(rel_fidelities[:, i_rep])*100:.2f}%', loc='left')
    # fig.savefig(DFUtils.create_filename(results_dir + rf'\{rep_rate}kHz_theta_cmap.pdf'))

cbar = fig.colorbar(pc, ax=axs.ravel().tolist())
cbar.set_label(r'$|\theta_{nm}|$')
fig.savefig(DFUtils.create_filename(results_dir + rf'\theta_cmap.pdf'))

'''Save costs and fidelities'''
np.save(DFUtils.create_filename(results_dir + rf'\optimal_least_squares.npy'), final_costs)

fidelities_df = pd.DataFrame(data=fidelities, index=indices, columns=[f'{f}kHz' for f in rep_vals])
fidelities_df.to_csv(results_dir + rf'\fidelities_with_ideal.csv')

rel_fid_df = pd.DataFrame(data=rel_fidelities, index=indices, columns=[f'{f}kHz' for f in rep_vals])
rel_fid_df.to_csv(results_dir + rf'\fidelities_with_{rep_vals[0]}kHz.csv')

'''Plot fidelities'''
x2 = np.arange(len(rep_vals))
y2 = np.arange(max_detected + 2)
X2, Y2 = np.meshgrid(x2, y2)

fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6), layout='constrained', sharex='all', sharey='all')

ax = axs2[0]
pc = ax.pcolormesh(X2, Y2, fidelities, vmin=0., vmax=1.)

ax.set_xlabel('Rep rate/kHz')
ax.set_ylabel(r'$n$')
ax.set_xticks(x2)
ax.set_xticklabels(rep_vals)
ax.set_yticks(y2)
ax.set_yticklabels(indices)
ax.set_title('(a) Fidelity with ideal', loc='left')

ax = axs2[1]
pc = ax.pcolormesh(X2, Y2, rel_fidelities, vmin=0., vmax=1.)
ax.set_xlabel('Rep rate/kHz')
ax.set_title('(b) Fidelity with 100kHz', loc='left')

cbar2 = fig2.colorbar(pc, ax=axs2.ravel().tolist())
cbar2.set_label(r'Fidelity')

fig2.savefig(results_dir + rf'\Fidelities.pdf')

plt.show()