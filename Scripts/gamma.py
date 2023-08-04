from src.utils import DataUtils
from src.traces import Traces
from src.ML_funcs import return_artifical_data
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit
from lmfit import Model, Parameter, report_fit
from scipy.stats import gamma
from tqdm.auto import tqdm
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


# actual_data = DataUtils.read_high_freq_data(frequency, power = 2, new=True, trigger=True)
# targetTraces = Traces(frequency=frequency, data=actual_data, multiplier=multiplier, num_bins=num_bins)
# offset_target = 0.5*targetTraces.return_av_diff()
# actual_data = actual_data - offset_target
#
data_high, label = return_artifical_data(500, 3)
# extracted_params = []
# guess = [700000, 4, 10]
#
# for i in tqdm(range(len(data_high))):
#     try:
#         y = data_high[i]
#         x = np.arange(1, len(y)+1)
#         fit, cov = curve_fit(gamma, x, y, p0 = guess, maxfev = 200000)
#         extracted_params.append(fit)
#     except (RuntimeError, OverflowError):
#         extracted_params.append([0,0,0])
# extracted_params = np.array(extracted_params)
#
# np.savetxt('extracted_params_gamma.txt', extracted_params)
#
extracted_params = []
for i in tqdm(range(len(data_high))):
    try:
        y = data_high[8]
        x = np.arange(0, len(y))
        guess = [-181860.9071243739, 25448.516309015893, 14.33975016573416, -26402750.986471906, 6.164476977344654, -1.9193093173505553e-05, 216398.60784343176, 11574548.485413479]
        fit, cov = curve_fit(exp_signal, x, y,p0 = guess,  maxfev = 8000000)

        extracted_params.append(np.array(list(fit)))
    except (RuntimeError, OverflowError):
        extracted_params.append([0,0,0,0,0,0,0,0])
extracted_params = np.array(extracted_params)

np.savetxt('extracted_params_exp.txt', extracted_params)
np.savetxt('labels.txt', label)
#for i in range(1):
# y = data_high[8]
# x = np.arange(0, len(y))
#
# fit, cov = curve_fit(exp_signal, x, y, maxfev=20000000)
# c, a, x0, t_rise, t_fall, d, e, k = fit
# plt.plot(y, label = 'data')
# plt.plot(x, exp_signal(x, c, a, x0, t_rise, t_fall,d, e, k), label = 'fit')
# plt.legend()
# plt.show()
