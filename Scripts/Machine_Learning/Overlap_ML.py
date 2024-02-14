import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial
import datetime
import time
import logging
import pandas as pd

from src.utils import DataUtils, LogUtils, DFUtils
from src.traces import Traces
from src.ML_funcs import ML, return_artifical_data, extract_features, find_offset

'''
script to produce  PN distributions using tabular classifiers, specify power, whether FE and modeltype.
'''
#specify parameters
power = 5
feature_extraction = False
modeltype = 'RF'
multiplier = 2
guess_mean_pn = 1.5

time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
results_dir = rf'..\..\Results\{modeltype}_ML_raw_{power}_{time_stamp}'

LogUtils.log_config(time_stamp='', dir=results_dir, filehead='log', module_name='', level=logging.INFO)
logging.info(f'Produce PN distributions using tabular classifiers. Raw data from raw_{power}, feature extraction is '
             f'{feature_extraction}, model type is {modeltype}. For inner product method at 100kHz, multiplier={multiplier}')

max_depth = 15
n_estimators = 600
logging.info(f'Classifier training uses max_depth={max_depth} (Only relevant for BDT model), '
             f'and n_estimators={n_estimators} (Only relevant for RF and BDT models).')

'''
define poisson distributions
'''

def poisson_curve(x, mu, A):
    return A * (mu ** x) * np.exp(-mu) / factorial(np.abs(x))

def poisson_norm(x, mu):
    return (mu ** x) * np.exp(-mu) / factorial(x)

'''
generate PN dist for 100kHz using overlap
'''
data100 = DataUtils.read_raw_data_new(100,power)
trace100 = Traces(100 , data100, multiplier)

x100,y = trace100.pn_bar_plot(plot = True, plt_title='100kHz bar plot')
plt.savefig(DFUtils.create_filename(results_dir + r'\100kHz_inner_prod_pn.pdf'))

dist100 = y/np.sum(y)

fit, cov = curve_fit(poisson_norm, x100, dist100, p0 = [guess_mean_pn]) #, maxfev=2000)
lam = fit[0]
lam_var = cov[0,0]

plt.figure('100kHz norm bar plot')
plt.bar(x100, dist100)

dense_x = np.linspace(min(x100), max(x100), 100)
plt.plot(dense_x, poisson_norm(dense_x, lam), color='red', label=rf'Poisson fit with $\mu={{{lam:.2f}}}$')

plt.xlabel('Photon number')
plt.ylabel('Probability')
plt.xticks(x100[::2])
plt.legend()
plt.title('100kHz normalised bar plot')
plt.savefig(results_dir + r'\100kHz_inner_prod_pn_norm.pdf')

logging.info(rf'Fit normalised Poisson distribution to 100kHz data: mu={lam}, cov={lam_var}.')


'''
ML for higher frequencies
'''

freq_values = np.arange(200,1001,100)
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# # results file by matthew, stored in scripts/params
# probabilities = np.zeros((len(freq_values)+1, len(dist100)))
# probabilities[0, :] = dist100

# results file by me, stored in results
results_df = pd.DataFrame(np.zeros((len(freq_values)+1, 4 + len(dist100))),
                          columns=['rep_rate', 'acc_score', 'fit_mu', 'fit_var'] + list(x100))
results_df.loc[0] = [100, np.nan, fit[0], cov[0,0]] + list(dist100)

i_freq = 0
for frequency,ax in zip(freq_values, axs.ravel()):
    i_freq = i_freq + 1
    data_high, labels = return_artifical_data(frequency, multiplier, power)
    features = data_high

    if feature_extraction:

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
    logging.info(f'Start building model for {frequency}kHz with artificial data')
    t1= time.time()
    model = ML(features, labels, modeltype=modeltype, max_depth=max_depth, n_estimators=n_estimators)
    model.makemodel()
    t2 = time.time()
    accuracy = model.accuracy_score()  # accuracy score of the test artificial samples (25% by default)
    logging.info(f'Model complete after {t2-t1}s. Test samples (25%) achieve accuracy score={accuracy}')

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

    #
    y_vals = np.bincount(predictions, minlength=len(x100))
    y_vals = y_vals / np.sum(y_vals)
    x_vals = np.arange(len(y_vals))

    ax.bar(x_vals, y_vals)
    ax.plot(x_vals, y_vals, 'kx')

    '''
    fit data
    '''
    fit, cov = curve_fit(poisson_norm, x_vals, y_vals, p0=[guess_mean_pn], maxfev = 2000)
    dense_x_vals = np.linspace(0, max(x_vals), 100)
    ax.plot(dense_x_vals, poisson_norm(dense_x_vals, fit[0]), label=rf'Poisson fit with $\mu={{{fit[0]:.2f}}}$', color='r')

    ax.set_title(f'{frequency}kHz accuracy score = {accuracy:.4f}')
    ax.legend()
    ax.set_xticks(x_vals[::2])

    '''
    Save results  
    '''
    # probabilities[i_freq] = y_vals
    results_df.loc[i_freq] = [frequency, accuracy, fit[0], cov[0,0]] + list(y_vals)
    results_df.to_csv(results_dir + rf'\{modeltype}_probs_raw{power}.csv', index=False)

    plt.savefig(results_dir + rf'\{modeltype}_probs_raw{power}.pdf')
# plt.show()

# '''
# ensure all PN dist are the same length for np.savetxt
# '''
#
# for l in probabilities:
#     while len(l)< len(probabilities[0]):
#         l.append(0)

'''
save probabilities for tomography
'''
results_df.to_csv(DFUtils.create_filename(rf'..\..\Params\{modeltype}_probs_raw{power}.csv'), index=False)

