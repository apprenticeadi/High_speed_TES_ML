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
ref_rep_rate = 100
high_rep_rates = np.arange(200, 1100, 100)
raw_traces_to_plot = 1000

sampling_rate = 5e4
dataReader = DataReader('Data/Tomography_data_2024_04')
data_group = 'power_0'

'''Inner product classifier'''
ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)

'''Read and label the reference data'''
ref_data = dataReader.read_raw_data(data_group, ref_rep_rate)
refTraces = Traces(ref_rep_rate, ref_data, parse_data=True, trigger_delay=0)

ipClassifier.train(refTraces)
ipClassifier.predict(refTraces, update=True)

'''Remove the baseline '''
# refTraces.pca_cleanup(1)  # pca clean up: bad idea
baseline_ref = refTraces.find_offset()
refTraces.data = refTraces.data - baseline_ref  # remove the baseline

# plt.figure('reference data flatten', figsize=(12, 5))
# plt.plot(refTraces.data[:10, :].flatten())
# plt.title('reference data')
# for i in range(10):
#     plt.axvline(i * 500, ymin=0, ymax=0.8, color='black', ls='dashed')
# plt.axhline(0, 0, 1, color='red', ls = 'dashed')

plt.figure('reference data')
for i in range(raw_traces_to_plot):
    plt.plot(refTraces.data[i, :], alpha=0.5)
plt.axhline(0, 0, 1, color='black', ls='dashed', alpha=0.5)

'''Loop over high rep rates'''
training_traces = {}
actual_traces = {}

plot_kwargs = {'figsize':(15, 10), 'sharex':True, 'sharey':True, 'layout':'constrained'}

fig1, axs1 = plt.subplot_mosaic(high_rep_rates.reshape((3,3)), **plot_kwargs)  # plot average trace
fig1.suptitle('Average trace')

fig2, axs2 = plt.subplot_mosaic(high_rep_rates.reshape((3,3)), **plot_kwargs)  # plot raw traces to see clustering
fig2.suptitle('Training traces')

fig3, axs3 = plt.subplot_mosaic(high_rep_rates.reshape((3,3)), **plot_kwargs)  # plot raw traces to see clustering
fig3.suptitle('Actual traces')
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

    '''Generate training '''
    trainingTraces = generate_training_traces(refTraces, high_rep_rate, trigger_delay=trigger_delay)

    # # for high overlap data, zero the zero-photon characteristic trace in training data.
    # if high_rep_rate >= 600:
    #     zero_of_the_zero = np.min(np.mean(trainingTraces.pn_traces(0), axis=0))
    #     if zero_of_the_zero >0:
    #         trainingTraces.data = trainingTraces.data - zero_of_the_zero

    trainingTraces.data = trainingTraces.data + baseline_ref  # add back the baseline
    training_traces[high_rep_rate] = trainingTraces

    '''Plot average traces'''
    ax1.set_title(f'{high_rep_rate}kHz')
    ax1.plot(trainingTraces.average_trace(), label='Training')
    ax1.plot(actualTraces.average_trace(), label='Actual')
    ax1.set_xlabel('Samples')

    '''Plot training traces'''
    ax2.set_title(f'{high_rep_rate}kHz')
    for i in range(raw_traces_to_plot):
        ax2.plot(trainingTraces.data[i], alpha=0.1)

    # plot characteristic traces
    char_trace_dict = trainingTraces.characteristic_traces()
    for pn in char_trace_dict.keys():
        ax2.plot(char_trace_dict[pn], color='red', ls='dashed')

    ax2.set_xlabel('Samples')

    '''Plot actual traces'''
    ax3.set_title(f'{high_rep_rate}kHz')
    for i in range(raw_traces_to_plot):
        ax3.plot(actualTraces.data[i], alpha=0.1)
    ax3.set_xlabel('Samples')

    # overlap training characteristic traces on top of the plot
    for pn in char_trace_dict.keys():
        ax3.plot(char_trace_dict[pn], color='red', ls='dashed')

axs1[high_rep_rates[0]].legend()
ax2.set_ylim([-1000, 25000])
ax3.set_ylim([-1000, 25000])
plt.show()

