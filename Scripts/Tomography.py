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
    # Define a custom colormap with more gradual transitions
    cmap_colors = [(1.0, 1.0, 0.0), (0.0, 0.0, 1.0)]  # Yellow to blue
    cmap = LinearSegmentedColormap.from_list('yellow_to_blue', cmap_colors)

    norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create the grid of squares with the custom colormap and normalization
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

    # Set the title
    plt.title(title)

    # Show the plot
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


rep_rate = 0
probs = [probabilities_raw5[rep_rate], probabilities_raw6[rep_rate], probabilities_raw7[rep_rate], probabilities_raw8[rep_rate]]

for i in range(4):
    while len(probs[i])<len(probs[1]):

        probs[i] = np.append(probs[i],[0])
probs = np.array(probs)

means = [1.34, 8.14, 6.03, 3.30]

def calculate_final_power(power_i, attenuation):

    power_f = power_i/(10**(attenuation/10))
    return power_f

pow = 3 # 0 = 1.34, 1 = 8.14, ...


def qmk(m_values, power=av_pn[pow][rep_rate], attenuation=attenuations[pow][rep_rate], reprate=rep_rates[pow][rep_rate]):
    f = 3e8 / (1550e-9)
    h = 6.6261e-34
    f_power = calculate_final_power(power * 10**-6, attenuation)
    alpha_k = np.sqrt((f_power / reprate) / (h * f))
    alpha_k = means[i]
    q_mk_values = np.exp(-(alpha_k**2)) * ((alpha_k**(2 * m_values)) / np.array([math.factorial(np.abs(m)) for m in m_values]))
    return np.array(q_mk_values)

qmk_vals = np.zeros((4,19))

for i in range(4):
    values = qmk(np.arange(0,19,1),power=av_pn[i][rep_rate], attenuation=attenuations[i][rep_rate], reprate=rep_rates[i][rep_rate])
    qmk_vals[i] = values


def cost(theta):
    theta = theta.reshape((17,19))
    c = 0
    for k in range(probs.shape[0]):
        for n, p in enumerate(probs[k]):
            if p!=0:
                c+= np.abs(p - np.sum(theta[n,:]*qmk_vals[k,:]))**2
    return c




bounds = [(0,1) for _ in range(17*19)]


guess = np.zeros((17,19))
np.fill_diagonal(guess, 1)
guess = guess.reshape(17*19)
t1 = time.time()
#results = minimize(cost, guess, bounds = bounds, method='SLSQP')
results = least_squares(cost, guess, bounds=(0,1))
t2 = time.time()
print('runtime = ' + str(t2-t1))
data = results.x
np.savetxt('Scripts/theta_vals.txt', data)
data = data.reshape((17,19))
create_color_plot(data, '100 kHz')



