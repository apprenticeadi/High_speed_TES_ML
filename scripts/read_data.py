import numpy as np
import matplotlib.pyplot as plt

from utils.data_reader import DataReader, RuquReader
from tes_resolver import Traces, TraceUtils,  DataChopper

'''Script that reads raw traces and plots them'''
rep_rate = 800
plot_trace_index_range = list(range(0,100))

# read coherent state data
dataReader = DataReader('Data/Tomography_data_2024_04')
data_group = 'power_6'

data_raw = dataReader.read_raw_data(data_group, rep_rate)

# # or read squeezed state data
# dataReader = RuquReader(r'Data/squeezed states 2024_07_17')
# data_keywords = [f'{rep_rate}kHz', '2024-07-17-2010_', '2nmPump', '1570nmBPF', 'Chan[1]']
#
# data_raw = dataReader.read_raw_data(*data_keywords, concatenate=True)

# trace object
rawTraces = Traces(rep_rate, data=data_raw, parse_data=False)

'''What the raw data looks like'''
plt.figure('raw data')
plt.plot(data_raw.flatten())
plt.xlim(min(plot_trace_index_range)*rawTraces.period, max(plot_trace_index_range) * rawTraces.period)

'''Chop the data'''
data_chopped = DataChopper.chop_traces(data_raw, samples_per_trace=rawTraces.period)
plt.figure('Chopped data without parsing')
for i in plot_trace_index_range:
    plt.plot(data_chopped[i], alpha=0.5)

'''Parse the data- their peak positions are more aligned with each other '''
data_parsed = TraceUtils.parse_data(rep_rate, data_raw, interpolated=False, trigger_delay=0)
plt.figure('Parsed data')
for i in plot_trace_index_range:
    plt.plot(data_parsed[i], alpha=0.5)

'''Parse the data with automatic triggering- Lose one trace per row '''
parsedTraces = Traces(rep_rate, data=data_raw, parse_data=True)
data_parsed_triggered = parsedTraces.data
plt.figure('Parsed data with trigger')
for i in plot_trace_index_range:
    plt.plot(data_parsed_triggered[i], alpha=0.5)

'''Average trace'''
average_trace = parsedTraces.average_trace()
plt.figure('Average trace')
plt.plot(average_trace)

plt.show()


