from scipy.special import factorial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
max_truncation = 17  #truncation photon number.

time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
results_dir = rf'..\Results\Tomography_data_2024_04\tomography_{time_stamp}'

params_dir = rf'..\Results\Tomography_data_2024_04\Params'
log_df = pd.read_csv(params_dir + r'\log_2024_04_22.csv')

LogUtils.log_config(time_stamp='', dir=results_dir, filehead='log', module_name='', level=logging.INFO)
logging.info(f'Run tomography on data from power_{powers}, processed with {modeltype} classifier. For tomography, '
             f'photon number truncated at max_truncation={max_truncation}. ')

rep_vals = np.arange(100, 1100, 100)

fig, axs = plt.subplots(2, 5, squeeze=True, sharey=True, sharex=True, layout='constrained', figsize=(15,7))
axs = axs.flatten()

final_costs = np.zeros(len(rep_vals))
fidelities = np.zeros(len(rep_vals))
for i_rep, rep_rate in enumerate(rep_vals):

    '''
    Load experimental probabilites- matrix P, with dimensions (N+1)*S, where N is the maximum photon number, S is the 
    number of powers. 
    P_ns is the experimental probability of measuring n photons from coherent light with power s=|alpha|^2
    '''
    probs = np.zeros((len(powers), max_truncation + 1))
    mean_pns = np.zeros(len(powers))
    max_photon = 0
    for i_power, power in enumerate(powers):
        mean_pns[i_power] = log_df.loc[(log_df['power_group'] == f'power_{power}') & (log_df['rep_rate/kHz'] == rep_rate), 'pm_estimated_av_pn'].iloc[0]

        df = pd.read_csv(params_dir + rf'\{modeltype}\{modeltype}_results_power_{power}.csv')
        pn = np.array(df.loc[df['rep_rate'] == rep_rate, '0':].iloc[0])
        if len(pn) > max_truncation + 1:
            pn = pn[:max_truncation + 1]
        probs[i_power, :len(pn)] = pn

        max_photon = max(max_photon, len(pn) - 1)
        # max_photon = 10

    probs = probs[:, :max_photon + 2]  # extra row in the end for pn = max_photon + 1 and beyond, experimental probs=0 for them
    probs = probs.T

    '''
    Calculate F matrix, dimension (M+1)*S, where M is the maximum truncation photon number. 
    F_ms is the theoretical (Poissonian) probability of measuring m photons from coherent light with power s=|alpha|^2. 
    '''
    F = np.zeros((len(powers), max_truncation + 1))
    ms = np.arange(max_truncation + 1)
    for i_power in range(len(powers)):
        F[i_power] = np.exp(- mean_pns[i_power]) * np.power(mean_pns[i_power], ms) / factorial(ms)
    F = F.T

    '''
    Theta is the POVM elements that we wish to find, dimension N+2 * M+1. 
    Theta_nm is the probability that the detector measures n photon, given m photon input. 
    n_th row corresponds to measuring n photons
    m-th column corresponds having m photons as input
    sum over row should be normalised to 1 
    sum over column should be normalised to 1 after normalisation by the last row. 
    '''
    guess_theta = np.zeros((max_photon + 2, max_truncation + 1))  # last row is for n=max_photon+1 and beyond

    '''
    define guess values and bounds
    '''
    # bounds = [(0,1)] * (max_photon+1) * max_truncation
    # for i in range(4, max_photon + 1):
    #     guess_theta[i, i] = 0.5
    #     guess_theta[i, i - 1] = 0.2
    #     guess_theta[i, i + 1] = 0.2
    #     guess_theta[i, i - 2] = 0.05
    #     guess_theta[i, i + 2] = 0.05
    np.fill_diagonal(guess_theta, 0.93)  # guess value
    guess_theta[0, 0] = 1.
    guess_theta[-1, -(max_truncation - max_photon):] = 1.
    guess_theta[0, 1] = 0.07  # given 1, measure 0
    for i in range(2, max_photon + 1):
        guess_theta[i - 2, i] = 0.004
        guess_theta[i - 1, i] = 0.066


    '''
    cvxpy least squares minimization, 
    cost function as a sum of squares
    constraints that sum over rows and columns equal 1
    '''
    theta = cp.Variable((max_photon + 2, max_truncation + 1), nonneg=True, value=guess_theta)
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

    # np.save(DFUtils.create_filename(results_dir + rf'\{rep_rate}kHz_theta.npy'), estimated_theta)

    indices = [f'{i}' for i in range(max_photon+1)] + [f'{max_photon+1}+']
    columns = [f'{i}' for i in range(max_truncation+1)]
    theta_df = pd.DataFrame(data=estimated_theta, index= indices, columns=list(range(max_truncation+1)))
    theta_df.to_csv(results_dir + rf'\{rep_rate}kHz_theta.csv')

    fidelity = np.trace(estimated_theta) / np.sum(estimated_theta)
    fidelities[i_rep] = fidelity

    '''
    create colour plot, using same blue to yellow colours as White paper
    '''
    # fig, ax = plt.subplots()
    ax = axs[i_rep]

    x = np.arange(estimated_theta.shape[1])
    y = np.arange(estimated_theta.shape[0])
    X, Y = np.meshgrid(x, y)

    pc = ax.pcolormesh(X, Y, estimated_theta,
                       norm=mcolors.SymLogNorm(linthresh=0.01))

    ax.set_xticks(x[::2])
    ax.set_xticklabels(columns[::2])
    ax.set_yticks(y)
    ax.set_yticklabels(indices)

    # ax.set_title(rf'({alphabet[i_rep]}) {rep_rate}kHz, fidelity={fidelity:.3g}, final cost={optimal_value:.2g}')
    ax.set_title(rf'({alphabet[i_rep]}) {rep_rate}kHz', loc='left')

    # fig.savefig(DFUtils.create_filename(results_dir + rf'\{rep_rate}kHz_theta_cmap.pdf'))


cbar = fig.colorbar(pc, ax=axs.ravel().tolist())
cbar.set_label(r'$|\theta_{nm}|$')
fig.savefig(DFUtils.create_filename(results_dir + rf'\theta_cmap.pdf'))

np.save(DFUtils.create_filename(results_dir + rf'\optimal_least_squares.npy'), final_costs)

fig2, axs2 = plt.subplots(1, 2, squeeze=True, sharex='all', figsize=(10, 4), layout='constrained')

ax = axs2[0]
ax.plot(rep_vals, fidelities, 'o-')
ax.set_ylim(0,1)
ax.set_xlabel('Rep rates/kHz')
ax.set_ylabel('Fidelity')
ax.set_title('(a) Fidelity', loc='left')

ax = axs2[1]
ax.plot(rep_vals, final_costs, 'o-')
ax.set_xlabel('Rep rates/kHz')
ax.set_ylabel('Optimised cost function value')
ax.set_title('(b) Final cost', loc='left')

fig2.savefig(results_dir + rf'\Fidelity_and_final_costs.pdf')

plt.show()