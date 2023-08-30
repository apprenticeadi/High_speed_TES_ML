import numpy as np
import math
from scipy.optimize import minimize
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cvxpy as cp
import time

def create_color_plot(data, title, figsize=(8, 6)):

    cmap_colors = [ (0.0, 0.0, 1.0),(1.0, 1.0, 0.0)]
    cmap = LinearSegmentedColormap.from_list('blue_to_yellow', cmap_colors)

    norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

    fig, ax = plt.subplots(figsize=figsize)

    cax = ax.matshow(data, cmap=cmap, norm=norm)

    # Add a colorbar
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('theta value')

    # Set axis labels and ticks
    ax.set_xlabel('m', size = 'xx-large')
    ax.set_ylabel('n', size = 'xx-large')

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(np.arange(0,len(data[0]),1))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(np.arange(0,len(data),1))

    plt.title(title)
    plt.show()
'''
load in data files, data is from log file
'''
log = np.loadtxt('Data/power_attenuation.txt', skiprows=1, unpack = True)


probabilities_raw5 = np.loadtxt('Scripts/raw_probabilities.txt', unpack = True).T
prob100_raw5 = np.loadtxt('Scripts/prob100.txt', unpack = True)
probabilities_raw5 = np.insert(probabilities_raw5,0,prob100_raw5,axis = 0)
probabilities_raw6 = np.loadtxt('Scripts/probabilities_raw6.txt', unpack = True).T
probabilities_raw7 = np.loadtxt('Scripts/probabilities_raw7.txt', unpack = True).T
probabilities_raw8 = np.loadtxt('Scripts/probabilities_raw8.txt', unpack = True).T

rep_rates = np.array_split(log[0]*10**3, 4)
av_pn =  np.array_split(log[1], 4)
attenuations = np.array_split(log[5], 4)
'''
remove poor distributions by k index for testing, uncomment if want to loop over all k
'''
delete = [1,2]
rep_rates = np.delete(rep_rates, delete, axis = 0)
av_pn = np.delete(av_pn, delete, axis = 0)
attenuations = np.delete(attenuations, delete, axis = 0)

rep_rate = 0 # 0=100kHz, 1 = 200kHz ...
probs = [probabilities_raw5[rep_rate], probabilities_raw6[rep_rate], probabilities_raw7[rep_rate], probabilities_raw8[rep_rate]]


'''removing poor distributions'''


probs = [probabilities_raw5[rep_rate], probabilities_raw8[rep_rate]]
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


means = np.array([1.1833,8.14,6.6, 1.83])

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
    f_power = calculate_final_power(power * 10**-6, attenuation)
    '''
    first alpha_k is the analytical calculation, second alpha_k uses the arbitrary values defined above, comment out for testing
    '''
    alpha_k = np.sqrt((f_power / reprate) / (h * f))
    alpha_k = means[i]
    q_mk_values = np.exp(-(alpha_k**2)) * ((alpha_k**(2 * m_values)) / np.array([math.factorial(np.abs(m)) for m in m_values]))
    return np.array(q_mk_values)

'''
find max photon number in sample
'''
lengths = []
for l in probs:
    lengths.append(len(l))
max_pn = max(lengths)
'''
define nmax and m
'''
nmax,m = max_pn, max_pn+1


'''
calculate qmk values
'''
qmk_vals = np.zeros((len(probs),m))

for i in range(len(probs)):
    values = qmk(np.arange(0,m,1),power=av_pn[i][rep_rate], attenuation=attenuations[i][rep_rate], reprate=rep_rates[i][rep_rate])
    qmk_vals[i] = values

'''
define cost function
'''
def cost(theta):
    theta = theta.reshape((nmax,m))
    c = 0
    for k in range(probs.shape[0]):
        for n, p in enumerate(probs[k]):
            if p!=0:
                c+= np.abs(p - np.sum(theta[n,:]*qmk_vals[k,:]))**2
    return c



'''
define guess values and bounds
'''
bounds = [(0,1) for _ in range(nmax*m)]
guess = np.zeros((nmax,m))
np.fill_diagonal(guess, 1)
guess = guess.reshape(nmax*m)
'''
perform least squares
'''
t1 = time.time()
#results = minimize(cost, guess, bounds = bounds, method='SLSQP') uncomment to try other minimization methods
results = least_squares(cost,guess, bounds=(0,1))
t2 = time.time()
print('runtime = ' + str(t2-t1))
data = results.x
np.savetxt('Scripts/theta_vals.txt', data)
data = data.reshape((nmax,m))

'''
calculate fidelity and create plot
'''
fidelity = np.trace(data)/np.sum(data)
print(fidelity)
create_color_plot(data, 'all k, 100 kHz')


# '''
# try cvxpy
# '''
# theta = cp.Variable((nmax,m), value = guess)
# cost = cp.sum_squares(cp.abs(probs - cp.matmul(qmk_vals, theta.T)))
#
# constraints = [0<=theta, theta <=1]
#
# problem = cp.Problem(cp.Minimize(cost), constraints)
#
# problem.solve()
# optimal_theta = theta.value
# create_color_plot(optimal_theta, 'cvxpy 100kHz')