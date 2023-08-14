from src.utils import DataUtils
from src.traces import Traces
from src.ML_funcs import return_artifical_data
import matplotlib.pyplot as plt
import numpy as np
#from FATS import FeatureSpace
import scipy
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
from multiprocessing import Pool
from numba import jit
frequency = 500
multiplier = 3
num_bins = 1000


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

def get_estimates(data):
    peak_loc = np.argmax(data)
    A = 0.5*max(data)
    return [1.65286440e+05, A,  peak_loc, -6.33323708e+11,3.61436807e+00,  3.36636145e-06, -2.33442580e+04, -1.12228310e+07]


data_high, label = return_artifical_data(200,2)
extracted_params = []
label_2 = []
error = 0
for i in tqdm(range(len(data_high))):
    try:
        y = data_high[i]
        x = np.arange(0, len(y))
        guess = [10, 240, 70, 40, 20,0.05, -20, -60]
        fit, cov = curve_fit(exp_signal, x, y,p0 = guess,  maxfev = 8000)
        extracted_params.append(list(fit))
        label_2.append(label[i])
    except (RuntimeError, OverflowError):
        error = error+1


# for i in tqdm(range(len(data_high))):
#     fs = FeatureSpace(data = data_high[i], featureList=None)
#     features = fs.calculate_features()
#     extracted_params.append(list(features.values()))


print(error)
extracted_params = np.array(extracted_params)

np.savetxt('extracted_exp.txt', extracted_params)
np.savetxt('labels.txt', label_2)

# y = data_high[17]
# x = np.arange(0, len(y))
# #guess = get_estimates(y)
# guess = [10, 240, 70, 40, 20,0.05, -20, -60]
# print('curve fitting ...')
# fit, cov = curve_fit(exp_signal, x, y,p0 = guess,  maxfev = 80000)
# c, a, x0, t_rise, t_fall, d, e, k = fit
# print(list(fit))
# fit, cov = curve_fit(exp_signal, x, y,p0 = fit,  maxfev = 80000)
# c, a, x0, t_rise, t_fall, d, e, k = fit
# plt.plot(y, label = 'data')
# plt.plot(x, exp_signal(x, c, a, x0, t_rise, t_fall, d, e, k), label = 'fit')
# plt.legend()
# plt.show()
