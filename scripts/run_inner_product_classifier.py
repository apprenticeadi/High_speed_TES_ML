import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
# import sys
#
# sys.path.insert(0, r'F:\TES_python')

from tes_resolver.classifier.inner_product import InnerProductClassifier
from tes_resolver.traces import Traces
from src.data_utils import DataReader
import tes_resolver.config as config


'''Data'''
dataReader = DataReader('Data_2024_04')

data_group = 'raw_2'
rep_rate = 100
file_num = 0
'''Attenuation calibration'''
attenuation_db = -66.2
attenuation_pctg = 10 ** (attenuation_db / 10)
laser_power = 176.7 / 1e9 / 1.03  #W
tes_input_power = laser_power * attenuation_pctg  # W
energy_per_pulse = tes_input_power / (rep_rate * 1000)
from scipy.constants import h, c

wavelength = 1550 * 1e-9

est_mean_ph = energy_per_pulse / (h * c / wavelength)
print(est_mean_ph)

'''Run the inner product classifier'''
data_raw = dataReader.read_raw_data(data_group, rep_rate, file_num=file_num)
calTraces = Traces(rep_rate, data_raw, parse_data=True, trigger=0)

ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)
ipClassifier.train(calTraces)

labels = ipClassifier.predict(calTraces, update=True)

pns, freq = calTraces.pn_distribution(normalised=True)

mean_photon_number = np.sum(pns * freq)
print(f'mean photon number is {mean_photon_number}')

plt.figure('Inner products')
overlaps = ipClassifier.calc_inner_prod(calTraces)
plt.hist(overlaps, bins=ipClassifier.num_bins, color='aquamarine')
for pn_bin in ipClassifier.inner_prod_bins.values():
    plt.axvline(pn_bin, ymin=0, ymax=0.5, ls='dashed')

plt.figure('Traces')
for i in range(1000):
    plt.plot(calTraces.data[i], alpha=0.1)
characeristic_traces = calTraces.characteristic_traces()
for pn in characeristic_traces.keys():
    plt.plot(characeristic_traces[pn], color='red', alpha=1.)

plt.figure('PN distribution')
plt.bar(pns, freq)

def poisson_norm(x, mu):
    return (mu ** x) * np.exp(-mu) / factorial(x)

plt.plot(pns, poisson_norm(pns, mean_photon_number), '-o', color='red', label=f'mean={mean_photon_number}')
plt.plot(pns, poisson_norm(pns, est_mean_ph), '-o', color='orange', label=f'mean={est_mean_ph}')
plt.legend()
#
# ave_trace = calTraces.average_trace()
# plt.plot(ave_trace, color='black', alpha=1.)


plt.show()

