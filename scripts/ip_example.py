import time

import numpy as np
import matplotlib.pyplot as plt

from tes_resolver.classifier.inner_product import InnerProductClassifier
from tes_resolver import Traces, DataChopper
from utils import DataReader, RuquReader

'''Script that runs inner product classifier on given data. Plots raw traces, stegosaurus, and photon number 
distribution'''
raw_traces_to_plot = 100

'''Read data'''
# coherent state data
dataReader = DataReader('Data/Tomography_data_2024_04')
data_group = 'power_6'
rep_rate = 100
file_num = 0
data_raw = dataReader.read_raw_data(data_group, rep_rate, file_num=file_num)

# # squeezed state data
# dataReader = RuquReader(r'Data/squeezed states 2024_07_17')
# rep_rate = 100
# data_keywords = [f'{rep_rate}kHz', '2024-07-17-1954_', '2nmPump', '1570nmBPF', 'Chan[1]']
# data_raw = dataReader.read_raw_data(*data_keywords, concatenate=True)

'''Parse data'''
targetTraces = Traces(rep_rate, data_raw, parse_data=True, trigger_delay='automatic')

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
for i in range(raw_traces_to_plot):
    ax.plot(targetTraces.data[i], alpha=0.1)
characeristic_traces = targetTraces.characteristic_traces()
for pn in characeristic_traces.keys():
    if pn == 0:
        ax.plot(characeristic_traces[pn], color='blue', alpha=0.5, label='Characteristic traces')
    else:
        ax.plot(characeristic_traces[pn], color='blue', alpha=0.5)
ax.set_xlabel('Samples')
ax.set_title(f'First {raw_traces_to_plot} raw traces')
ave_trace = targetTraces.average_trace()
ax.plot(ave_trace, color='black', alpha=1., label='Average trace')
ax.legend()

ax = axs[2]
ax.set_title('PN distribution')
ax.bar(pns, freq)
ax.set_xlabel('Photon number')
#

#

plt.show()

