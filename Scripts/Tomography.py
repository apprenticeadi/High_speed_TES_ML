import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cvxpy as cp
'''
script to perform a tomography routine on the probabilities, using the cvxpy package
'''
extra_attenuation = 2.9
modeltype = 'CNN'
'''
load in data files, data is from log file, probabilities are calculated in different script and saved in params file
'''
log = np.loadtxt('Data/power_attenuation.txt', skiprows=1, unpack = True)


probabilities_raw5 = np.loadtxt(rf'Scripts/params/{modeltype}_probs_raw5.txt', unpack = True).T
probabilities_raw6 = np.loadtxt(rf'Scripts/params/C{modeltype}_probs_raw5.txtt', unpack = True).T
probabilities_raw7 = np.loadtxt(rf'Scripts/params/{modeltype}_probs_raw5.txt', unpack = True).T
probabilities_raw8 = np.loadtxt(rf'Scripts/params/{modeltype}_probs_raw5.txt', unpack = True).T


rep_rates = np.array_split(log[0]*10**3, 4)
av_pn =  np.array_split(log[1], 4)
attenuations = np.array_split(log[5], 4)
av_pn[1], av_pn[2], av_pn[3] = av_pn[1]*10, av_pn[2]*10, av_pn[3]*10
'''
remove poor distributions by k index for testing, uncomment if want to loop over all k
'''
# delete = [1]
# rep_rates = np.delete(rep_rates, delete, axis = 0)
# av_pn = np.delete(av_pn, delete, axis = 0)
# attenuations = np.delete(attenuations, delete, axis = 0)


rep_vals = np.arange(0,9.1,1)
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
fid_100 = []
for rep,ax in zip(rep_vals, axs.ravel()):

    rep_rate = int(rep) # 0=100kHz, 1 = 200kHz ...
    probs = [probabilities_raw5[rep_rate], probabilities_raw6[rep_rate], probabilities_raw7[rep_rate], probabilities_raw8[rep_rate]]

    '''removing poor distributions if want to  '''

    #probs = [probabilities_raw5[rep_rate], probabilities_raw7[rep_rate],probabilities_raw8[rep_rate]]
    '''
    ensure all probabilities are the same length, fill lower powers wil 0 values for higher PN
    '''
    if len(probs) ==1:
        num=0
    else:
        num=1
    for i in range(len(probs)):
        while len(probs[i])<len(probs[num]):
            probs[i] = np.append(probs[i],[0])
    probs = np.array(probs)
    '''
    ensure same amount of powers for probabilities and parameters
    '''
    if len(rep_rates) != len(probs):
        raise 'k mismatch'

    def calculate_final_power(power_i, attenuation):
        '''
        function to calculate the output power after attenuation
        '''
        power_f = power_i/(10**(attenuation/10))
        return power_f

    pow = 0 # 0 = 1.34, 1 = 8.14, ...

    def qmk(m_values, power=av_pn[pow][rep_rate], attenuation=attenuations[pow][rep_rate], reprate=rep_rates[pow][rep_rate]):
        f = 3e8 / (1550e-9)
        h = 6.6261e-34
        f_power = calculate_final_power(power * 10**-6, attenuation+extra_attenuation)
        '''
        calculate alpha_k
        '''
        alpha_k = np.sqrt((f_power / reprate) / (h * f))
        q_mk_values = np.exp(-(alpha_k**2)) * ((alpha_k**(2 * m_values)) / np.array([math.factorial(np.abs(m)) for m in m_values]))
        return np.array(q_mk_values)

    '''
    find max photon number in sample
    '''
    max_pn = len(max(probs, key = lambda x:len(x)))
    '''
    define nmax and m
    '''
    nmax,m = max_pn, max_pn

    '''
    calculate qmk values
    '''
    qmk_vals = np.zeros((len(probs),m))

    for i in range(len(probs)):
        values = qmk(np.arange(0,m,1),power=av_pn[0][rep_rate], attenuation=attenuations[i][rep_rate], reprate=rep_rates[i][rep_rate])
        qmk_vals[i] = values
    '''
    define guess values and bounds
    '''
    bounds = [(0,1) for _ in range(nmax*m)]
    guess = np.zeros((nmax,m))
    np.fill_diagonal(guess, 1)

    '''
    cvxpy least squares minimization, theta as a nxm matrix
    cost function as a sum of squares
    constraints that sum over rows and columns equal 1
    fidelity calculated as trace/sum
    '''
    theta = cp.Variable((nmax, m), nonneg=True, value=guess)
    cost = cp.sum_squares(cp.abs(probs - cp.matmul(qmk_vals, theta)))
    constraints = [0 <= theta,theta <=1, cp.sum(theta, axis=0) == 1, cp.sum(theta, axis = 1)==1]
    problem = cp.Problem(cp.Minimize(cost), constraints)

    problem.solve()

    estimated_theta = theta.value

    fidelity = np.trace(estimated_theta)/ np.sum(estimated_theta)
    if rep_rate ==0:
        theta100 = estimated_theta
    else:
        fidelity100 = 1 - np.sum(np.abs(estimated_theta-theta100))/ np.sum(theta100)
        fid_100.append(fidelity100)
    '''
    create colour plot, using same blue to yellow colours as white paper
    '''
    cmap_colors = [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
    cmap = LinearSegmentedColormap.from_list('blue_to_yellow', cmap_colors)
    norm = mcolors.Normalize(vmin=np.min(estimated_theta), vmax=np.max(estimated_theta))
    cax = ax.matshow(estimated_theta, cmap=cmap, norm=norm)

    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('theta value')

    ax.set_xlabel('m', size = 'xx-large')
    ax.set_ylabel('n', size = 'xx-large')

    ax.set_xticks(np.arange(estimated_theta.shape[1]))
    ax.set_xticklabels(np.arange(0,len(estimated_theta[0]),1))
    ax.set_yticks(np.arange(estimated_theta.shape[0]))
    ax.set_yticklabels(np.arange(0,len(estimated_theta),1))

    ax.set_title(fr'least squares, {rep_rates[0][rep_rate]/1000} kHz, fidelity = {fidelity:.4f}')

plt.tight_layout()
plt.show()

freq_values = np.arange(200,901,100)
plt.plot(freq_values, fid_100, '+')
plt.xlabel('repitition rate', size = 'x-large')
plt.ylabel(r'adjusted fidelity , 1 - $\frac{\sum_{i,j} \theta^{100}_{i,j} - \theta^{rr}_{i}}{\sum_{i,j}\theta^{100}_{i,j}}$', size = 'x-large')
plt.tight_layout()
plt.show()

