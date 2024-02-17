import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial
import datetime
import time
import logging
import pandas as pd

from src.utils import LogUtils, DFUtils
from src.data_utils import DataParser, DataChopper
from src.traces import Traces
from src.ML_funcs import ML, extract_features #return_artifical_data, find_offset

'''
script to produce  PN distributions using tabular classifiers, specify power, whether FE and modeltype.
'''
#specify parameters
data_dir = 'RawData'
sub_power_name = 'raw_6'
feature_extraction = False
modeltype = 'RF'
test_size = 0.1
multiplier = 1.
num_bins = 1000
guess_mean_pn = 3
vertical_shift = True
triggered = True
pca_components = None

dataParser = DataParser(sub_dir=sub_power_name, parent_dir=data_dir)


time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
results_dir = rf'..\..\Results\{modeltype}\{sub_power_name}_{time_stamp}'

LogUtils.log_config(time_stamp='', dir=results_dir, filehead='log', module_name='', level=logging.INFO)
logging.info(rf'Produce PN distributions using tabular classifiers. Raw data from {data_dir}\{sub_power_name}, ' 
             f'feature extraction is {feature_extraction}, model type is {modeltype}. \n'
             f'Test_size = {test_size}.\n'
             f'Inner product method used to identify calibration data at 100kHz with multiplier={multiplier} and num_bins={num_bins}. \n'
             f'The 100kHz calibration data is used to generate training data by overlapping them to higher frequencies. \n'
             f'pca_components={pca_components}- pca is used to clean up the 100kHz data to generate training data. '
             f'vertical_shift={vertical_shift}- this is whether the training data is shifted to meet the same average-trace height as the actual data.\n'
             f'Trigger={triggered}- this is whether theactual data are  triggered on the rising edge of traces (only for more than 300kHz). \n'
             f'Training data is shifted horizontally to match peaks with actual data'
             )

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
Load calibration traces 
'''
data100 = dataParser.parse_data(100, interpolated=False, triggered=False)
refTraces = Traces(100, data100, multiplier, num_bins=num_bins)

# run pca on calibration traces if required.
if pca_components is None:
    calTraces = refTraces
else:
    # use pca cleaned-up traces as calibration traces.
    data_pca = refTraces.pca_cleanup(num_components=pca_components)
    calTraces = Traces(100, data_pca, multiplier, num_bins=num_bins)

calTraces.fit_histogram(plot=True)
plt.savefig(DFUtils.create_filename(results_dir + rf'\100kHz_fit_stegosaurus.pdf'))

'''
Process calibration traces
'''
t1 = time.time()
_, counts = calTraces.pn_bar_plot(plot = True, plt_title='100kHz bar plot')  # unnormalised
t2 = time.time()
cal_time = t2 -t1

logging.info(rf'{cal_time}s to produce photon number plot for 100kHz data')

plt.savefig(DFUtils.create_filename(results_dir + fr'\100kHz_inner_prod_pn.pdf'))

pns, ref_distrib = calTraces.pn_bar_plot(plot=True, plt_title='100kHz photon number distribution', normalised=True)  # normalised

'''
Fit Poissonian distribution
'''
fit, cov = curve_fit(poisson_norm, pns, ref_distrib, p0 = [guess_mean_pn]) #, maxfev=2000)
lam = fit[0]
lam_var = cov[0,0]

plt.plot(pns, poisson_norm(pns, lam), color='red', marker='x', linestyle='solid', label=rf'$\mu={{{lam:.4f}}}$')

plt.xlabel('Photon number')
plt.ylabel('Probability')
plt.xticks(pns[::2])
plt.legend()
plt.title('100kHz normalised bar plot')
plt.savefig(results_dir + r'\100kHz_inner_prod_pn_norm.pdf')

logging.info(rf'Fit normalised Poisson distribution to 100kHz data: mu={lam}, cov={lam_var}.')

'''
ML for higher frequencies
'''
freq_values = np.arange(200,1001,100)
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))  # prediction results
fig2, axs2 = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))  # average trace
fig3, axs3 = plt.subplots(nrows=3, ncols=3, figsize=(15,12))  # raw trace
# # results file by matthew, stored in scripts/params
# probabilities = np.zeros((len(freq_values)+1, len(dist100)))
# probabilities[0, :] = dist100

# results file by me, stored in results
results_df = pd.DataFrame(columns=['rep_rate', 'acc_score', 'fit_mu', 'fit_var', 'training_t', 'predict_t'] + list(pns))
results_df.loc[0] = [100, np.nan, lam, lam_var, np.nan, cal_time] + list(ref_distrib)

i_freq = 0
for frequency in freq_values:
    i_freq = i_freq + 1

    '''
    Load in real data
    '''
    actual_data = dataParser.parse_data(frequency, interpolated=False, triggered=triggered)  # DataUtils.read_high_freq_data(frequency, power= power, new= True)
    av_actual = np.mean(actual_data, axis=0)
    period = actual_data.shape[1]

    '''
    Generate training data by overlapping the calibration traces 
    '''
    training_data, training_labels = calTraces.generate_training_data(frequency)  # untriggered.
    av_training = np.mean(training_data, axis=0)
    assert training_data.shape[1] == period

    '''
    Correct for vertical and horizontal shift
    '''
    h_offset = np.argmax(av_training) - np.argmax(av_actual)
    if h_offset < 0:
        h_offset = h_offset + period

    training_data, training_labels = DataChopper.chop_labelled_traces(training_data.flatten(), training_labels,
                                                                      period, trigger=h_offset)

    if vertical_shift:
        v_offset = np.max(np.mean(training_data, axis=0)) - np.max(av_actual)
        training_data = training_data - v_offset

    '''
    Plot raw traces
    '''
    ax3 = axs3.ravel()[i_freq - 1]
    ax3.plot(training_data[10:30].flatten(), label='training')
    ax3.plot(actual_data[10:30].flatten(), label='actual')
    if i_freq == 1:
        ax3.legend()
    fig3.savefig(results_dir + rf'\training_vs_actual.pdf')

    '''
    Plot average traces
    '''
    ax2 = axs2.ravel()[i_freq - 1]
    ax2.plot(av_training, linestyle='dotted', label='training')
    ax2.plot(av_actual, linestyle='solid', label='actual')
    ax2.plot(np.mean(training_data, axis=0), linestyle='dashed', label='shifted training')

    ax2.set_title(f'{frequency}kHz')
    if i_freq == 1:
        ax2.legend()
    fig2.savefig(results_dir + rf'\training_vs_actual_av_trace.pdf')

    '''
    Feature extraction on training data
    '''
    features = training_data
    if feature_extraction:

        '''
        extract features, should really implement this before but as currently updating the function keep here for now
        '''

        peak_data = []
        for series in training_data:
            feature = extract_features(series)
            peak_data.append(feature)

        features = np.array(peak_data)

    '''
    Feature extraction on actual data 
    '''
    actual_features = actual_data
    if feature_extraction == True:
        actual_data_features = []
        for series in actual_data:
            feature = extract_features(series)
            actual_data_features.append(feature)

        actual_features = np.array(actual_data_features)


    '''
    Build model
    '''
    logging.info(f'Start building model for {frequency}kHz with artificial data')
    t1= time.time()
    model = ML(features, training_labels, modeltype=modeltype, max_depth=max_depth, n_estimators=n_estimators,
               test_size=test_size)
    model.makemodel()
    t2 = time.time()

    training_t = t2 -t1
    accuracy = model.accuracy_score()  # accuracy score of the test artificial samples (25% by default)
    logging.info(f'Model complete after {training_t}s. Test samples ({test_size*100}%) achieve accuracy score={accuracy}')

    '''
    make predictions
    '''
    t1 = time.time()
    predictions = model.predict((actual_features))
    t2 = time.time()
    predict_t = t2 - t1
    logging.info(f'Actual data predicted after {predict_t}s.')

    #
    y_vals = np.bincount(predictions, minlength=len(pns))
    y_vals = y_vals / np.sum(y_vals)
    x_vals = np.arange(len(y_vals))

    ax = axs.ravel()[i_freq - 1]
    ax.bar(x_vals, y_vals)
    # ax.plot(x_vals, y_vals, 'kx')

    '''
    fit data
    '''
    fit, cov = curve_fit(poisson_norm, x_vals, y_vals, p0=[guess_mean_pn], maxfev = 2000)
    # ax.plot(x_vals, poisson_norm(x_vals, fit[0]), label=rf'$\mu={{{fit[0]:.2f}}}$', color='r')  # plot curve fit
    ax.plot(x_vals, poisson_norm(x_vals, lam), color='red', marker='x', linestyle='solid', label=rf'$\mu={{{lam:.4f}}}$')  # plot 100kHz distribution

    ax.set_title(f'{frequency}kHz')
    ax.legend()
    ax.set_xticks(x_vals[::2])

    '''
    Save results  
    '''
    # probabilities[i_freq] = y_vals
    results_df.loc[i_freq] = [frequency, accuracy, fit[0], cov[0,0], training_t, predict_t] + list(y_vals)
    results_df.to_csv(results_dir + rf'\{modeltype}_results_{sub_power_name}.csv', index=False)

    fig.savefig(results_dir + rf'\{modeltype}_results_{sub_power_name}.pdf')
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
# results_df.to_csv(DFUtils.create_filename(rf'..\..\Params\{modeltype}_probs_raw{power}.csv'), index=False)

