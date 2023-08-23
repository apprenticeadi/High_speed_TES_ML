import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from src.ML_funcs import ML, return_artifical_data, extract_features, find_offset
from scipy.optimize import curve_fit
from scipy.special import factorial
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

'''
script to produce PN distributions using a tabular classifier
'''
power = 5
feature_extraction = False


def poisson_curve(x, mu, A):
    return A * (mu ** x) * np.exp(-mu) / factorial(np.abs(x))


def poisson_norm(x, mu):
    return (mu ** x) * np.exp(-mu) / factorial(np.abs(x))


data100 = DataUtils.read_raw_data_new(100, power)
trace100 = Traces(100, data100, 1.8)
x, y = trace100.pn_bar_plot(plot=False)
fit, cov = curve_fit(poisson_norm, x, y / np.sum(y), p0=[5], maxfev=2000)
prob100 = y/np.sum(y)
print(prob100)
np.savetxt('prob100.txt', prob100)
lam = fit[0]

# freq_values = np.arange(200, 1001, 100)
#
# chi_vals = []
# probabilities = []
#
# for frequency in freq_values:
#     print(frequency)
#     data_high, filtered_label = return_artifical_data(frequency, 2, power)
#     features = data_high
#     if feature_extraction == True:
#         peak_data = []
#         for series in data_high:
#             feature = extract_features(series)
#             peak_data.append(feature)
#
#         features = np.array(peak_data)
#     X_train, X_test, y_train, y_test = train_test_split(features, filtered_label, test_size=0.2, random_state=42)
#
#     rf_model = RandomForestClassifier(n_estimators=300)
#     rf_model.fit(X_train, y_train)
#     print('models_built')
#     actual_data = DataUtils.read_high_freq_data(frequency, power=power, new=True)
#     shift = find_offset(frequency, power)
#     actual_data = actual_data - shift
#     actual_features = actual_data
#
#     if feature_extraction == True:
#         actual_data_features = []
#         for series in actual_data:
#             feature = extract_features(series)
#             actual_data_features.append(feature)
#
#         actual_features = np.array(actual_data_features)
#
#
#     test = rf_model.predict((actual_features))
#
#     y_vals = np.bincount(test) / np.sum(np.bincount(test))
#     x_vals = list(range(len(y_vals)))
#     probabilities.append(y_vals)
#     expected = poisson_norm(x_vals, lam)
#
#     chisq = []
#     for i in range(len(expected)):
#         chi = ((expected[i] - y_vals[i]) ** 2) / expected[i]
#         chisq.append((chi))
#     chi_vals.append(sum(chisq))
#
#
# print(chi_vals)
# np.savetxt('chi_vals_ML.txt', chi_vals)
# chi_vals = np.array(chi_vals)
# print(chi_vals)
# plt.legend()
# plt.show()







