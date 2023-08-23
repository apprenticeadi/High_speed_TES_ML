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
Algorithm :
1.load probability distributions and power data
2.define key functions
3.perform least squares algorithm to find theta
'''

data = np.loadtxt('Data/power_attenuation.txt', skiprows=1, unpack = True)
probabilities_raw5 = np.loadtxt('Scripts/raw_probabilities.txt', unpack = True)
probabilities = probabilities_raw5.T
prob100 = np.loadtxt('Scripts/prob100.txt', unpack = True)

reprate, power, attenuation = data[0]*10**3, data[1], data[5]

def calculate_final_power(power_i, attenuation):

    power_f = power_i/(10**(attenuation/10))
    return power_f
rep = 0
def qmk(m, power = power[rep], attenuation = attenuation[rep], reprate = reprate[rep]*10**3):
    f = 3e8/(1550e-9)
    h = 6.6261e-34
    f_power = calculate_final_power(power*10**-6, attenuation)
    #alpha_k = np.sqrt((f_power/reprate)/(h*f))
    alpha_k = np.sqrt(1.7)
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
for i in range(len(probabilities[rep])):
    def function(t):
        p =  probabilities[rep+1][i]
        return abs(p - theta(*t))**2

    # Initial guess for t0...t20
    initial_guess = (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Perform the optimization
    #result = minimize(function, initial_guess, method='BFGS')
    result = least_squares(function, initial_guess)
    optimized_values = result.x
    m_vals.append(optimized_values[0:7])
m_vals = np.array(m_vals)


create_color_plot(m_vals, str(reprate[rep]/1000) + 'kHz, mean = 1.34')





