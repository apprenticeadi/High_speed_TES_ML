import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.constants import h, c

# import sys
#
# sys.path.insert(0, r'F:\TES_python')

from tes_resolver.classifier.inner_product import InnerProductClassifier
from tes_resolver.traces import Traces
from src.data_reader import DataReader
import tes_resolver.config as config


'''Data'''
dataReader = DataReader('Data_2024_04')

data_group = 'raw_20'
rep_rate = 100
file_num = 0
raw_traces_to_plot = 1000

'''Estimate mean photon number '''
attenuation_db = -60.16
attenuation_pctg = 10 ** (attenuation_db / 10)
bs_ratio = 98.35526316
bs_ratio_error = 0.73315
# laser_power = 1842 / 1e9 / 1.0138  #W
pm_reading = 3.668 * 1e-6  # W
pm_reading_error = 0.001 * 1e-6
wavelength = 1550 * 1e-9

laser_power = pm_reading / bs_ratio  #W
tes_input_power = laser_power * attenuation_pctg  # W
energy_per_pulse = tes_input_power / (rep_rate * 1000)
est_mean_ph = energy_per_pulse / (h * c / wavelength)
est_error = est_mean_ph * np.sqrt(bs_ratio_error**2/ bs_ratio**2 + pm_reading_error**2/ pm_reading**2)
print(rf'Estimated mean photon number = {est_mean_ph} with error= {est_error}')

'''Run the inner product classifier'''
data_raw = dataReader.read_raw_data(data_group, rep_rate, file_num=file_num)
calTraces = Traces(rep_rate, data_raw, parse_data=True, trigger=0)

ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)
ipClassifier.train(calTraces)

labels = ipClassifier.predict(calTraces, update=True)

pns, freq = calTraces.pn_distribution(normalised=True)

mean_photon_number = np.sum(pns * freq)
print(f'TES reported mean photon number is {mean_photon_number}')

fig, axs = plt.subplots(1, 3, figsize=(18, 5), layout='constrained')

ax = axs[0]
ax.set_title('Inner products')
overlaps = ipClassifier.calc_inner_prod(calTraces)
ax.hist(overlaps, bins=ipClassifier.num_bins, color='aquamarine')
for pn_bin in ipClassifier.inner_prod_bins.values():
    ax.axvline(pn_bin, ymin=0, ymax=0.5, ls='dashed')
ax.set_xlabel('Inner product')
ax.set_ylabel('Counts')

ax = axs[1]
for i in range(raw_traces_to_plot):
    ax.plot(calTraces.data[i], alpha=0.1)
characeristic_traces = calTraces.characteristic_traces()
for pn in characeristic_traces.keys():
    ax.plot(characeristic_traces[pn], color='red', alpha=1., label='Characteristic traces')
ax.set_xlabel('Samples')
ax.set_title(f'First {raw_traces_to_plot} raw traces')

ax = axs[2]
ax.set_title('PN distribution')
ax.bar(pns, freq)
ax.set_xlabel('Photon number')
def poisson_norm(x, mu):
    return (mu ** x) * np.exp(-mu) / factorial(x)

ax.plot(pns, poisson_norm(pns, mean_photon_number), '-o', color='red', label=f'mean={mean_photon_number}')
ax.plot(pns, poisson_norm(pns, est_mean_ph), '-o', color='orange', label=f'mean={est_mean_ph}')
ax.legend()
#
# ave_trace = calTraces.average_trace()
# plt.plot(ave_trace, color='black', alpha=1.)


plt.show()

