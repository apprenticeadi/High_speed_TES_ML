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

from src.utils import LogUtils, DFUtils

#TODO: for higher rep rates, the convergence by cvxpy is just bad, but this makes the final thetas look ok. How to account for this? Read Humphreys again.

'''
script to perform a tomography routine on the probabilities, using the cvxpy package
'''
# extra_attenuation = 2.9
modeltype = 'RF'

powers = [5, 8, 7]
max_truncation = 20  #truncation photon number.

time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
results_dir = rf'..\Results\Tomography\raw_{powers}_{time_stamp}'

LogUtils.log_config(time_stamp='', dir=results_dir, filehead='log', module_name='', level=logging.INFO)
logging.info(rf'Run tomography on data from raw_{powers}, with photon number truncated at max_truncation={max_truncation}. '
             )

rep_vals = np.arange(100, 1100, 100)
final_costs = np.zeros(len(rep_vals))
for i_rep, rep_rate in enumerate(rep_vals):

    '''
    Load experimental probabilites- matrix P, with dimensions (N+1)*S, where N is the maximum photon number, S is the 
    number of powers. 
    P_ns is the experimental probability of measuring n photons from coherent light with power s=|alpha|^2
    '''
    probs = np.zeros((len(powers), max_truncation+1))
    mean_pns = np.zeros(len(powers))
    max_photon = 0
    for i_power, power in enumerate(powers):
        df = pd.read_csv(rf'params\{modeltype}_results_raw_{power}.csv')
        mean_pns[i_power] = df.loc[df['rep_rate']==rep_rate, 'fit_mu'].iloc[0]
        pn = np.array(df.loc[df['rep_rate']==rep_rate, '0':].iloc[0])
        probs[i_power, :len(pn)] = pn

        max_photon = max(max_photon, len(pn)-1)

    probs = probs[:, :max_photon+1]
    probs = probs.T

    '''
    Calculate F matrix, dimension (M+1)*S, where M is the maximum truncation photon number. 
    F_ms is the theoretical (Poissonian) probability of measuring m photons from coherent light with power s=|alpha|^2. 
    '''
    F = np.zeros((len(powers), max_truncation+1))
    ms = np.arange(max_truncation+1)
    for i_power in range(len(powers)):
        F[i_power] = np.exp(- mean_pns[i_power]) * np.power(mean_pns[i_power], ms) / factorial(ms)
    F = F.T

    '''
    Theta is the POVM elements that we wish to find, dimension N+1 * M+1. 
    Theta_nm is the probability that the detector measures n photon, given m photon input. 
    n_th row corresponds to measuring n photons
    m-th column corresponds having m photons as input
    sum over row should be normalised to 1 
    sum over column should be smaller than 1.
    '''
    guess_theta = np.zeros((max_photon+1, max_truncation+1))
    np.fill_diagonal(guess_theta, 1)  # guess value
    '''
    define guess values and bounds
    '''
    # bounds = [(0,1)] * (max_photon+1) * max_truncation


    '''
    cvxpy least squares minimization, 
    cost function as a sum of squares
    constraints that sum over rows and columns equal 1
    '''
    theta = cp.Variable((max_photon+1, max_truncation+1), nonneg=True, value=guess_theta)
    cost = cp.sum_squares(cp.abs(probs - theta @ F))
    constraints = [0 <= theta, theta <=1, cp.sum(theta, axis=0) <= 1, cp.sum(theta, axis = 1)==1]
    problem = cp.Problem(cp.Minimize(cost), constraints)

    t1 = time.time()
    optimal_value = problem.solve()
    t2 = time.time()
    msg= f'For {rep_rate}kHz, min squares routine finish after {t2-t1}s, optimal_value = {optimal_value}'
    logging.info(msg)
    final_costs[i_rep] = optimal_value

    estimated_theta = theta.value

    np.save(DFUtils.create_filename(results_dir + rf'\{rep_rate}kHz_theta.npy'), estimated_theta)

    fidelity = np.trace(estimated_theta)/ np.sum(estimated_theta)

    '''
    create colour plot, using same blue to yellow colours as white paper
    '''
    fig, ax = plt.subplots(1,1, squeeze=True)
    cmap_colors = [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
    cmap = LinearSegmentedColormap.from_list('blue_to_yellow', cmap_colors)
    norm = mcolors.Normalize(vmin=np.min(estimated_theta), vmax=np.max(estimated_theta))
    cax = ax.matshow(estimated_theta, cmap=cmap, norm=norm)

    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('theta value')

    ax.set_xlabel('m', size = 'xx-large')
    ax.set_ylabel('n', size = 'xx-large')

    # ax.set_xticks(np.arange(0, estimated_theta.shape[1], 2))
    # ax.set_xticklabels(np.arange(0,len(estimated_theta[0]),1))
    # ax.set_yticks(np.arange(estimated_theta.shape[0]))
    # ax.set_yticklabels(np.arange(0,len(estimated_theta),1))
    #
    ax.set_title(fr'{rep_rate}kHz, fidelity = {fidelity:.4f}')
    fig.savefig(results_dir+rf'\{rep_rate}kHz_theta_cmap.pdf')

