import numpy as np
import math
from scipy.optimize import minimize

'''
Algorithm :
1.load probability distributions and power data
2.define key functions
3.perform least squares algorithm to find theta
'''

data = np.loadtxt('Data/power_attenuation.txt', skiprows=1, unpack = True)
probabilities_raw5 = np.loadtxt('Scripts/raw_probabilities.txt', unpack = True)
probabilities = probabilities_raw5.T

reprate, power, attenuation = data[0], data[1], data[5]

def calculate_final_power(power_i, attenuation):

    power_f = power_i/(10**(attenuation/10))

    return power_f

def qmk(m, power, attenuation, pulse = 70e-9):
    f = 3e8/(1550e-9)
    h = 6.6261e-34

    f_power = calculate_final_power(power*10**-6, attenuation)
    alpha_k = np.sqrt((f_power*pulse)/(h*f))
    q_mk = np.exp(-(alpha_k**2))* ((alpha_k**(2*m))/math.factorial(m))

    return q_mk
M=3
m_vals = np.linspace(0,M,M+1)
q_vals = []
for m in m_vals:
    q_vals.append(qmk(m,power[0], attenuation[0]))
print(np.sum(q_vals))





