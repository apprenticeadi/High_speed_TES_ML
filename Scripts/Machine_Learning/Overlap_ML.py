import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from src.ML_funcs import ML, return_artifical_data, extract_features
from scipy.optimize import curve_fit
from scipy.special import factorial

power = 7

def poisson_curve(x, mu, A):
    return A * (mu ** x) * np.exp(-mu) / factorial(np.abs(x))

freq_values = np.arange(200,1001,100)
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
av_PN = []
av_PN_error = []
chi_vals = []

for frequency,ax in zip(freq_values, axs.ravel()):
    print(frequency)
    data_high, filtered_label = return_artifical_data(frequency,2,power)
    peak_data = []
    for series in data_high:
        feature = extract_features(series)
        peak_data.append(feature)

    features = np.array(peak_data)
    model = ML(features, filtered_label, modeltype='RF')
    model.makemodel(num_rounds=25)

    actual_data = DataUtils.read_high_freq_data(frequency, power= power, new= True)

    art_trace, actual_trace = Traces(frequency,data_high,2), Traces(frequency,actual_data,2)
    av1,a,b = actual_trace.average_trace()
    av2,c,d = art_trace.average_trace()
    shift = np.max(av1) - np.max(av2)
    actual_data = actual_data - shift


    actual_data_features = []
    for series in actual_data:
        feature = extract_features(series)
        actual_data_features.append(feature)


    actual_features = np.array(actual_data_features)
    test = model.predict((actual_features))


    y_vals = np.bincount(test)
    x_vals = list(range(len(y_vals)))
    ax.bar(x_vals, y_vals)
    ax.plot(x_vals, y_vals, 'x')

    x_vals = np.array(x_vals)

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
np.savetxt('chi_vals_ML.txt', chi_vals)
plt.plot(freq_values, chi_vals, '+')
plt.xlabel('frequency')
plt.ylabel('chi-square')
plt.show()