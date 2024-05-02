import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.constants import h, c

from tes_resolver.classifier.inner_product import InnerProductClassifier
from tes_resolver.traces import Traces
from tes_resolver.data_chopper import DataChopper
from src.data_reader import DataReader
import tes_resolver.config as config
from src.utils import DFUtils, poisson_norm, estimate_av_pn

'''Script that runs inner product classifier on given data. Plots raw traces, stegosaurus, and photon number 
distribution'''

'''Data'''
dataReader = DataReader('Data/Tomography_data_2024_04')

data_group = 'power_0'
rep_rate = 500
file_num = 0
raw_traces_to_plot = 1000

'''Estimate mean photon number '''
attenuation_db = -60.16
bs_ratio = 98.35526316
bs_ratio_error = 0.73315
pm_reading = 8.710 * 1e-6  # W
pm_reading_error = 0.01 * 1e-6

est_mean_ph, est_error = TomoUtils.estimate_av_pn(rep_rate, pm_reading, attenuation_db, bs_ratio, pm_error=pm_reading_error, bs_error=bs_ratio_error)
print(rf'Estimated mean photon number = {est_mean_ph} with error= {est_error}')

'''Read data '''
data_raw = dataReader.read_raw_data(data_group, rep_rate, file_num=file_num)

if rep_rate <= 300:
    trigger_delay = 0
else:
    trigger_delay = DataChopper.find_trigger(data_raw, samples_per_trace=int(5e4 / rep_rate))

targetTraces = Traces(rep_rate, data_raw, parse_data=True, trigger_delay=0)

'''Run the inner product classifier'''
ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)
ipClassifier.train(targetTraces)

t1 = time.time()
labels = ipClassifier.predict(targetTraces, update=True)
t2 = time.time()
print(f'ip classifier predicts {targetTraces.num_traces} in {t2-t1}s')

pns, freq = targetTraces.pn_distribution(normalised=True)

mean_photon_number = np.sum(pns * freq)
print(f'TES reported mean photon number is {mean_photon_number}')

fig, axs = plt.subplots(1, 3, figsize=(18, 5), layout='constrained')

ax = axs[0]
ax.set_title('Inner products')
overlaps = ipClassifier.calc_inner_prod(targetTraces)
ax.hist(overlaps, bins=ipClassifier.num_bins, color='aquamarine')
for pn_bin in ipClassifier.inner_prod_bins.values():
    ax.axvline(pn_bin, ymin=0, ymax=0.5, ls='dashed')
ax.set_xlabel('Inner product')
ax.set_ylabel('Counts')

ax = axs[1]
# for i in range(raw_traces_to_plot):
#     ax.plot(targetTraces.data[i], alpha=0.1)
# characeristic_traces = targetTraces.characteristic_traces()
# for pn in characeristic_traces.keys():
#     ax.plot(characeristic_traces[pn], color='red', alpha=1., label='Characteristic traces')
ax.set_xlabel('Samples')
ax.set_title(f'First {raw_traces_to_plot} raw traces')

ax = axs[2]
ax.set_title('PN distribution')
ax.bar(pns, freq)
ax.set_xlabel('Photon number')

ax.plot(pns, poisson_norm(pns, mean_photon_number), '-o', color='red', label=f'mean={mean_photon_number}')
ax.plot(pns, poisson_norm(pns, est_mean_ph), '-o', color='orange', label=f'mean={est_mean_ph}')
ax.legend()
#
# ave_trace = targetTraces.average_trace()
# plt.plot(ave_trace, color='black', alpha=1.)
#
fig2, ax2 = plt.subplots(figsize=(20, 8))
ax2.plot(targetTraces.data[:100000, :].flatten())
ax2.set_xlabel('Samples')

plt.show()

