import numpy as np
import pandas as pd
import os

from utils import DataReader, DFUtils
from tes_resolver.traces import Traces, PCATraces
from tes_resolver.classifier import InnerProductClassifier

'''Plot the pca data from tabular classification'''

'''Which data to use'''
dataReader = DataReader(r'\Data\Tomography_data_2024_04')
data_group = 'power_6'

rep_rate = 500

save_dir = rf'../../Plots/Tomography_data_2024_04/trace_pca_plots/{data_group}/{rep_rate}kHz'
os.makedirs(save_dir, exist_ok=True)

'''Parameters'''
sampling_rate = 5e4
card_range_pm = 1  # card range is +/- 1V
voltage_precision = 2 * card_range_pm / (np.power(2, 14)-1)  # unit V

num_traces = 1000

'''Save raw traces parsed into voltage values'''
data_raw = dataReader.read_raw_data(data_group, rep_rate=rep_rate)
curTraces = Traces(rep_rate, data_raw, parse_data=True, trigger_delay='automatic')

data_parsed = curTraces.data
data_to_plot = data_parsed[:2 * num_traces].reshape((num_traces, 2 * curTraces.period))
data_to_plot = data_to_plot / 4 * voltage_precision * 1000

x = np.arange(data_to_plot.shape[1]) / sampling_rate * 1000  # unit us

df = pd.DataFrame(data_to_plot.T, columns=np.arange(num_traces))
df.insert(loc=0, column='time/us', value=x)

df.to_csv(save_dir + rf'\first_{num_traces}traces.csv', index=False)

'''Save inner product stegosaurus'''
ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)
ipClassifier.train(curTraces)

# get stegosaurus
overlaps = ipClassifier.calc_inner_prod(curTraces)
heights, bin_edges = np.histogram(overlaps, bins=ipClassifier.num_bins)
heights = np.append(heights, np.nan)
hist_df = pd.DataFrame({'num_traces': heights, 'ip_bin_edges': bin_edges})

hist_df.to_csv(save_dir + rf'\ip_stegosaurus.csv', index=False)


