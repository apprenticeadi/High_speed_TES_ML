import numpy as np
import math
from scipy.optimize import minimize
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


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

probs = [probabilities_raw5, probabilities_raw6, probabilities_raw7, probabilities_raw8]
means = [1.34, 8.14, 6.03, 3.30]

def calculate_final_power(power_i, attenuation):

    power_f = power_i/(10**(attenuation/10))
    return power_f


rep = 0 #0 = 100kHz, 1 = 200kHz, ...
int = 0 # 0 = 1.34, 1 = 8.14, ...

def qmk(m, power = av_pn[int][rep], attenuation = attenuations[int][rep], reprate = rep_rates[int][rep]):
    f = 3e8/(1550e-9)
    h = 6.6261e-34
    f_power = calculate_final_power(power*10**-6, attenuation)
    #f_power = f_power*0.9 # other loss
    alpha_k = np.sqrt((f_power/reprate)/(h*f))
    #alpha_k = np.sqrt(1.34)
    q_mk = np.exp(-(alpha_k**2))* ((alpha_k**(2*m))/math.factorial(m))
    return q_mk



def theta(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20):
    result = (
            t0 * qmk(0) +
            t1 * qmk(1) +
            t2 * qmk(2) +
            t3 * qmk(3) +
            t4 * qmk(4) +
            t5 * qmk(5) +
            t6 * qmk(6) +
            t7 * qmk(7) +
            t8 * qmk(8) +
            t9 * qmk(9) +
            t10 * qmk(10) +
            t11 * qmk(11) +
            t12 * qmk(12) +
            t13 * qmk(13) +
            t14 * qmk(14) +
            t15 * qmk(15) +
            t16 * qmk(16) +
            t17 * qmk(17) +
            t18 * qmk(18) +
            t19 * qmk(19) +
            t20 * qmk(20)
    )

    return result


m_vals = []

for i in range(len(probs[int][rep])):
    def function(t):
        p =  probs[int][rep][i]
        return abs(p - theta(*t))**2

    # Initial guess for t0...t20
    initial_guess = (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Perform the optimization
    #result = minimize(function, initial_guess, method='BFGS')
    result = least_squares(function, initial_guess)
    optimized_values = result.x
    m_vals.append(optimized_values[0:len(probs[int][0])])
m_vals = np.array(m_vals)


create_color_plot(m_vals, str(rep_rates[int][rep]/1000) + r'kHz, mean =' +str(means[int]))





