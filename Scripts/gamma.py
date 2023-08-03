from src.utils import DataUtils
from src.traces import Traces
from src.ML_funcs import return_artifical_data
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.stats import gamma
from tqdm.auto import tqdm
frequency = 500
multiplier = 3
num_bins = 1000

def gamma(x,A, k, theta):
    return A*(1 / (theta**k * np.math.gamma(k))) * (x**(k - 1)) * np.exp(-x / theta)


actual_data = DataUtils.read_high_freq_data(frequency, power = 2, new=True, trigger=True)
targetTraces = Traces(frequency=frequency, data=actual_data, multiplier=multiplier, num_bins=num_bins)
offset_target = 0.5*targetTraces.return_av_diff()
actual_data = actual_data - offset_target

data_high, label = return_artifical_data(500, 3)
extracted_params = []
guess = [700000, 4, 10]

for i in tqdm(range(len(data_high))):
    try:
        y = data_high[i]
        x = np.arange(1, len(y)+1)
        fit, cov = curve_fit(gamma, x, y, p0 = guess, maxfev = 200000)
        extracted_params.append(fit)
    except (RuntimeError, OverflowError):
        extracted_params.append([0,0,0])
extracted_params = np.array(extracted_params)

np.savetxt('extracted_params.txt', extracted_params)

