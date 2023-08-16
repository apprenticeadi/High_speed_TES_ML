from src.utils import DataUtils
from src.traces import Traces
from src.ML_funcs import return_artifical_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

frequency, multiplier, numbins = 500,3,1000

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


data_high, label = return_artifical_data(500,1.8,5)
# extracted_params = []
# label_2 = []
# error = 0
# for i in tqdm(range(len(data_high))):
#     try:
#         y = data_high[i]
#         x = np.arange(0, len(y))
#         guess = [-44588.952333487876, -6797.467649918665, 36.44751873035673, 9.2579839028286, -638443.0428272152, -4.4961548964486484e-05, 75546.12590605592, 1338605.2674221238]
#         fit, cov = curve_fit(exp_signal, x, y,p0 = guess,  maxfev = 10000)
#         extracted_params.append(list(fit))
#         label_2.append(label[i])
#     except (RuntimeError, OverflowError):
#         error = error+1
#
# print(error)
# extracted_params = np.array(extracted_params)
#
# np.savetxt('extracted_exp.txt', extracted_params)
# np.savetxt('labels.txt', label_2)


y = data_high[0]
x = np.arange(0, len(y))
guess = [-44588.952333487876, -6797.467649918665, 36.44751873035673, 9.2579839028286, -638443.0428272152, -4.4961548964486484e-05, 75546.12590605592, 1338605.2674221238]
fit, cov = curve_fit(exp_signal, x, y,p0=guess, maxfev = 2000)
c, a, x0, t_rise, t_fall,d, e, k = fit
print(list(fit))
plt.plot(y, '+')
x = np.linspace(-20,100,1500)
plt.plot(x, exp_signal(x,c, a, x0, t_rise, t_fall,d, e, k))
plt.show()