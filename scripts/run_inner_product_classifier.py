import numpy as np
import matplotlib.pyplot as plt

from tes_resolver.classifier.inner_product import InnerProductClassifier
from tes_resolver.traces import Traces
from src.data_utils import DataReader
import tes_resolver.config as config

'''Run the inner product classifier'''

dataReader = DataReader('data_20240410')

data_group = 'raw_1'
rep_rate = 100

data_raw = dataReader.read_raw_data(data_group, rep_rate)
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

#
# ave_trace = calTraces.average_trace()
# plt.plot(ave_trace, color='black', alpha=1.)


plt.show()

attenuation_db = -87
attenuation_pctg = 10 ** (attenuation_db / 10)

laser_power = 13.15 / 1e6 / 1.005  #W
tes_input_power = laser_power * attenuation_pctg  # W

energy_per_pulse = tes_input_power / (rep_rate * 1000)

from scipy.constants import h, c

wavelength = 1550 * 1e-9

est_mean_ph = energy_per_pulse / (h * c / wavelength)
print(est_mean_ph)