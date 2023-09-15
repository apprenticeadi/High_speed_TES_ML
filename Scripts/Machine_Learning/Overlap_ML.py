import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils
from src.traces import Traces
from src.ML_funcs import ML, return_artifical_data, extract_features, find_offset
from scipy.optimize import curve_fit
from scipy.special import factorial

'''
script to produce PN distributions using tabular classifiers, specify power, whether FE and modeltype.
'''
#specify parameters
power = 7
feature_extraction = False
modeltype = 'RF'

'''
define poisson distributions
'''

def poisson_curve(x, mu, A):
    return A * (mu ** x) * np.exp(-mu) / factorial(np.abs(x))

def poisson_norm(x,mu):
    return (mu ** x) * np.exp(-mu) / factorial(np.abs(x))

'''
generate PN dist for 100kHz using overlap
'''

data100 = DataUtils.read_raw_data_new(100,power)
trace100 = Traces(100 , data100, 1.8)

x,y = trace100.pn_bar_plot(plot = False)
fit, cov = curve_fit(poisson_norm,x,y/np.sum(y), p0 = [8], maxfev=2000)
lam = fit[0]

freq_values = np.arange(200,1001,100)
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

dist100 = y/np.sum(y)
probabilities = [dist100]

for frequency,ax in zip(freq_values, axs.ravel()):

    data_high, labels = return_artifical_data(frequency,2,power)
    features = data_high

    if feature_extraction==True:

        '''
        extract features, should really implement this before but as currently updating the function keep here for now
        '''

        peak_data = []
        for series in data_high:
            feature = extract_features(series)
            peak_data.append(feature)

        features = np.array(peak_data)

    '''
    build model
    '''

    model = ML(features, labels, modeltype=modeltype, max_depth=15, n_estimators=600)
    model.makemodel()

    '''
    load in real data
    '''

    actual_data = DataUtils.read_high_freq_data(frequency, power= power, new= True)
    shift = find_offset(frequency, power)
    actual_data = actual_data - shift
    actual_features = actual_data

    if feature_extraction==True:
        actual_data_features = []
        for series in actual_data:
            feature = extract_features(series)
            actual_data_features.append(feature)

        actual_features = np.array(actual_data_features)

    '''
    make predictions
    '''

    predictions = model.predict((actual_features))

    y_vals = np.bincount(predictions)/np.sum(np.bincount(predictions))
    x_vals = list(range(len(y_vals)))

    probabilities.append(list(y_vals))

    ax.bar(x_vals, y_vals)
    ax.plot(x_vals, y_vals, 'x')

    x_vals = np.array(x_vals)
    '''
    fit data
    '''
    fit, cov = curve_fit(poisson_curve, x_vals, y_vals, p0=[3, np.sum(y_vals)], maxfev = 2000)
    x = np.linspace(0,max(x_vals),100)
    ax.plot(x, poisson_curve(x, fit[0], fit[1]) , label = 'poisson fit', color = 'r')

    expected = poisson_norm(x_vals, lam)# useful if calculating chi square vals

    ax.set_title(f'{frequency}kHz accuracy score = {model.accuracy_score():.4f}')
    ax.legend()

plt.show()

'''
ensure all PN dist are the same length for np.savetxt
'''

for l in probabilities:
    while len(l)< len(probabilities[0]):
        l.append(0)

'''
save probabilities for tomography
'''

np.savetxt(f'params/{modeltype}_probs_raw{power}.txt', probabilities)

