import numpy as np
import matplotlib.pyplot as plt
import time

from tes_resolver.ml_funcs import generate_training_traces
from tes_resolver.traces import Traces
from tes_resolver.classifier import InnerProductClassifier, TabularClassifier
from tes_resolver.data_chopper import DataChopper
from src.data_reader import DataReader

# overlap the 100kHz data, and compare the average trace with the actual high rep_rate data. How big is the vertical shift?

# maybe also try plotting the raw traces and see how they cluster (training data vs actual data)

'''Parameters'''
ref_rep_rate = 200
high_rep_rates = np.arange(300, 1100, 100)
raw_traces_to_plot = 1000

sampling_rate = 5e4
dataReader = DataReader('Tomography_data_2024_04')
data_group = 'power_6'

'''Inner product classifier'''
ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)

# '''Find the vertical offset'''
# cal_data_raw = dataReader.read_raw_data(data_group, ref_rep_rate)
# calTraces = Traces(ref_rep_rate, cal_data_raw, parse_data=True, trigger=0)
#
# ipClassifier.train(calTraces)
# ipClassifier.predict(calTraces, update=True)
#
# vertical_offset = calTraces.find_offset()

'''Loop over high rep rates'''
training_traces = {}
actual_traces = {}

fig1, axs1 = plt.subplot_mosaic(high_rep_rates.reshape((2,4)), figsize=(15, 6), sharex=True, sharey=True, layout='constrained')  # plot average trace
fig1.suptitle('Average trace')

fig2, axs2 = plt.subplot_mosaic(high_rep_rates.reshape((2,4)), figsize=(15, 6), sharex=True, sharey=True, layout='constrained')  # plot raw traces to see clustering
fig2.suptitle('Raw training traces')

fig3, axs3 = plt.subplot_mosaic(high_rep_rates.reshape((2,4)), figsize=(15, 6), sharex=True, sharey=True, layout='constrained')  # plot raw traces to see clustering
fig3.suptitle('Raw actual traces')
for high_rep_rate in high_rep_rates:
    ax1 = axs1[high_rep_rate]
    ax2 = axs2[high_rep_rate]
    ax3 = axs3[high_rep_rate]

    '''Read actual traces'''
    actual_data = dataReader.read_raw_data(data_group, high_rep_rate)

    if high_rep_rate <= 300:
        trigger_delay = 0
    else:
        trigger_delay = DataChopper.find_trigger(actual_data, samples_per_trace=int(sampling_rate/high_rep_rate))

    actualTraces = Traces(high_rep_rate, actual_data, parse_data=True, trigger_delay=trigger_delay)
    actual_traces[high_rep_rate] = actualTraces

    '''Read reference traces with the correct trigger'''
    ref_data = dataReader.read_raw_data(data_group, ref_rep_rate)
    refTraces = Traces(ref_rep_rate, ref_data, parse_data=True, trigger_delay=trigger_delay)

    '''Label reference traces'''
    ipClassifier.train(refTraces)  # retrain the ip classifier
    ipClassifier.predict(refTraces, update=True)
    vertical_offset = refTraces.find_offset()  # this is the average value of zero-photon traces

    '''Generate training data'''
    trainingTraces = generate_training_traces(refTraces, high_rep_rate)

    num_tails = int(high_rep_rate / ref_rep_rate + 0.5) - 1
    trainingTraces.data = trainingTraces.data - num_tails * vertical_offset  # subtract the appropriate number of offsets.
    training_traces[high_rep_rate] = trainingTraces

    '''Plot average traces'''
    ax1.set_title(f'{high_rep_rate}kHz')
    ax1.plot(trainingTraces.average_trace(), label='Training')
    ax1.plot(actualTraces.average_trace(), label='Actual')
    ax1.set_xlabel('Samples')

    '''Plot raw traces to see clustering'''
    ax2.set_title(f'{high_rep_rate}kHz')
    for i in range(raw_traces_to_plot):
        ax2.plot(trainingTraces.data[i], alpha=0.1)
    ax2.set_xlabel('Samples')

    ax3.set_title(f'{high_rep_rate}kHz')
    for i in range(raw_traces_to_plot):
        ax3.plot(actualTraces.data[i], alpha=0.1)
    ax3.set_xlabel('Samples')

axs1[high_rep_rates[0]].legend()
ax2.set_ylim([-1000, 25000])
ax3.set_ylim([-1000, 25000])
plt.show()

