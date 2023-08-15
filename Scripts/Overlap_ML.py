import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from src.ML_funcs import ML
from scipy.stats import poisson
from scipy.optimize import curve_fit
from scipy.special import factorial
from tqdm.auto import tqdm
multiplier = 1.8
num_bins = 1000
guess_peak = 30
pca_components = 2  # it's really doubtful if pca helps at all
pca_cleanup = True

power = 5

data_100 = DataUtils.read_raw_data_new(100,power)
calibrationTraces = Traces(frequency=100, data=data_100, multiplier=multiplier, num_bins=num_bins)
labels = calibrationTraces.return_labelled_traces()
filtered_ind = np.where(labels == -1)[0]
filtered_traces = np.delete(data_100, filtered_ind, axis = 0)
#filtered_traces = filtered_traces - calibrationTraces.return_av_diff()
filtered_label = np.delete(labels, filtered_ind)
print(np.bincount(filtered_label))
print(str(100*((len(data_100) - len(filtered_label))/len(data_100))) +'% filtered')
filtered_data = Traces(100, filtered_traces)


def poisson_curve(x, mu, A):
    return A * (mu ** x) * np.exp(-mu) / factorial(np.abs(x))

frequency = 800
freq_values = np.arange(200,1001,100)
#freq_values = [700,700,700,700,700,700,700,700,700]
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
av_PN = []
av_PN_error = []
chi_vals = []
shift = [1000,1000,1000,1000,700,800,1500,1600,1750]
j = -1
for frequency,ax in zip(freq_values, axs.ravel()):
    j = j+1
    print(j)
    data_high = filtered_data.overlap_to_high_freq(frequency)

    model = ML(data_high, filtered_label, modeltype='RF')
    model.makemodel(num_rounds=25)
    actual_data = DataUtils.read_high_freq_data(frequency, power= power, new= True)
    targetTraces = Traces(frequency=frequency, data=actual_data, multiplier=multiplier, num_bins=num_bins)


    offset_target = 0.5*targetTraces.return_av_diff()
    if frequency <550:
        offset_target, _ = targetTraces.subtract_offset()
    if frequency > 551:
        offset_target = 0.5 * targetTraces.return_av_diff()
        offset_target = shift[j]
    actual_data = actual_data - offset_target
    if pca_cleanup:
        actualTraces = Traces(frequency=frequency, data=actual_data)
        actual_data = actualTraces.pca_cleanup(num_components=pca_components)

    test = model.predict((actual_data))


    x_vals = list(range(len(np.bincount(test))))
    y_vals = np.bincount(test)
    ax.bar(x_vals, y_vals)
    ax.plot(x_vals, y_vals, 'x')

    x_vals = np.array(x_vals)

        #return poisson.pmf(x,mu)

    fit, cov = curve_fit(poisson_curve, x_vals, y_vals, p0=[1.5, np.sum(y_vals)], maxfev = 2000)
    av_PN.append(fit[0])
    av_PN_error.append(np.sqrt(cov[0,0]))

    x = np.linspace(0,max(x_vals),100)
    ax.plot(x, poisson_curve(x, fit[0], fit[1]) , label = 'poisson fit', color = 'r')

    expected = poisson_curve(x_vals, fit[0], fit[1])

    chisq = []
    for i in range(len(expected)):
        chi = ((expected[i] - y_vals[i]) ** 2) / expected[i]
        chisq.append((chi))
    chi_vals.append(sum(chisq))
    ax.set_title(str(frequency)+ 'kHz accuracy score = '+str(model.accuracy_score())[0:5])
    ax.legend()

plt.show()

plt.errorbar(freq_values, av_PN, yerr = av_PN_error,fmt = 'o', capsize=3)
plt.xlabel('frequency (kHz)')
plt.ylabel(r'lambda from fit')
plt.show()
ind = np.argmin(np.array(chi_vals))
print(ind, shift[ind])
np.savetxt('chi_vals_ML.txt', chi_vals)
plt.plot(freq_values, chi_vals, '+')
plt.xlabel('frequency')
plt.ylabel('chi-square')
plt.show()