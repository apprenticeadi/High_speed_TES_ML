from src.ML_funcs import return_artifical_data
import numpy as np
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
'''
script to find the fitting parameters for traces (INCOMPLETE)
'''
frequency = 500
power = 5

def gamma(x,A, k, theta):
    return A*(1 / (theta**k * np.math.gamma(k))) * (x**(k - 1)) * np.exp(-x / theta)

def exp_signal(x, c, a, x0, t_rise, t_fall,d, e, k):
    return c*np.exp(-d*(x+e)) + a *( (np.exp((x-x0)/t_rise) + np.exp((x0-x)/t_fall))**-1) +k

def crystal_ball(x, alpha, beta, m, sigma, A, c):
    z = (x - m) / sigma
    abs_alpha = np.abs(alpha)
    abs_z = np.abs(z)

    A = (beta / abs_alpha) ** alpha * np.exp(-0.5 * abs_alpha ** 2)
    B = abs_alpha / abs_z

    condition = (z > -alpha)
    pdf_values = np.where(condition, np.exp(-0.5 * z ** 2) / sigma, A * (B - abs_alpha / abs_z) ** (-alpha))

    return A*pdf_values + c


data_high, label = return_artifical_data(frequency, 1.8, power)
extracted_params = []
label_2 = []
error = 0

for i in tqdm(range(len(data_high))):
    try:
        y = data_high[i]
        x = np.arange(0, len(y))
        fit, cov = curve_fit(exp_signal, x, y,   maxfev = 10000)
        extracted_params.append(list(fit))
        label_2.append(label[i])
    except (RuntimeError, OverflowError):
        error = error+1

print(f'number of fitting errors:{error}')
extracted_params = np.array(extracted_params)

np.savetxt('params/extracted_exp.txt', extracted_params)
np.savetxt('params/extracted_labels.txt', label_2)

