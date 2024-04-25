import numpy as np
import matplotlib.pyplot as plt

from src.data_reader import DataReader
from tes_resolver.traces import Traces, TraceUtils
from tes_resolver.data_chopper import DataChopper

'''Script that reads raw traces and plots them'''

dataReader = DataReader('RawData')

rep_rate = 600
data_raw = dataReader.read_raw_data('raw_5', rep_rate=rep_rate)

rawTraces = Traces(rep_rate, data=data_raw, parse_data=False)

plot_trace_index_range = list(range(0,100))

'''What the raw data looks like'''
plt.figure('raw data')
plt.plot(data_raw[0])
plt.xlim(min(plot_trace_index_range)*rawTraces.period, max(plot_trace_index_range) * rawTraces.period)


'''Chop the data'''
data_chopped = DataChopper.chop_traces(data_raw, samples_per_trace=rawTraces.period)
plt.figure('Chopped data without parsing')
for i in plot_trace_index_range:
    plt.plot(data_chopped[i], alpha=0.5)

'''Parse the data- their peak positions are more aligned with each other '''
data_parsed, _ = TraceUtils.parse_data(rep_rate, data_raw, interpolated=False, trigger=0)
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


