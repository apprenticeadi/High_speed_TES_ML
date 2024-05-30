import numpy as np
import matplotlib.pyplot as plt
import time

from tes_resolver.ml_funcs import generate_training_traces
from tes_resolver.traces import Traces
from tes_resolver.classifier import InnerProductClassifier
from tes_resolver.data_chopper import DataChopper
from src.data_reader import DataReader

# overlap the 100kHz data, and compare the average trace with the actual high rep_rate data. How big is the vertical shift?

# maybe also try plotting the raw traces and see how they cluster (training data vs actual data)

'''Parameters'''
cal_rep_rate = 100
high_rep_rates = np.arange(200, 1100, 100)
raw_traces_to_plot = 1000

sampling_rate = 5e4
dataReader = DataReader('Data/Tomography_data_2024_04')
data_group = 'power_0'

'''Inner product classifier'''
ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)

'''Read and label the reference data'''
cal_data = dataReader.read_raw_data(data_group, cal_rep_rate)
calTraces = Traces(cal_rep_rate, cal_data, parse_data=True, trigger_delay=0)

ipClassifier.train(calTraces)
ipClassifier.predict(calTraces, update=True)

'''Remove the baseline '''
# refTraces.pca_cleanup(1)  # pca clean up: bad idea
cal_baseline = calTraces.find_offset()
calTraces.data = calTraces.data - cal_baseline  # remove the baseline

# plt.figure('reference data flatten', figsize=(12, 5))
# plt.plot(refTraces.data[:10, :].flatten())
# plt.title('reference data')
# for i in range(10):
#     plt.axvline(i * 500, ymin=0, ymax=0.8, color='black', ls='dashed')
# plt.axhline(0, 0, 1, color='red', ls = 'dashed')

t1 = time.time()
plt.figure('calibration data')
for i in range(raw_traces_to_plot):
    plt.plot(calTraces.data[i, :], alpha=0.5)
plt.axhline(0, 0, 1, color='black', ls='dashed', alpha=0.5)
t2 = time.time()
print(f'Time to plot {raw_traces_to_plot} traces from calibration data is {t2-t1}s')

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
    ti = time.time()
    trainingTraces = generate_training_traces(calTraces, high_rep_rate, trigger_delay=trigger_delay)

    # # for high overlap data, zero the zero-photon characteristic trace in training data.
    # if high_rep_rate >= 600:
    #     zero_of_the_zero = np.min(np.mean(trainingTraces.pn_traces(0), axis=0))
    #     if zero_of_the_zero >0:
    #         trainingTraces.data = trainingTraces.data - zero_of_the_zero

    # trainingTraces.data = trainingTraces.data + baseline_ref  # add back the baseline

    # correct for the vertical shift
    offset = np.max(trainingTraces.average_trace()) - np.max(actualTraces.average_trace())
    trainingTraces.data = trainingTraces.data - offset

    tf = time.time()
    print(f'Generate training traces for {high_rep_rate}kHz took {tf - ti}s')

    training_traces[high_rep_rate] = trainingTraces

    char_trace_dict = trainingTraces.characteristic_traces()

    '''Plot average traces'''
    ax1.set_title(f'{high_rep_rate}kHz')
    ax1.plot(trainingTraces.average_trace(), label='Training')
    ax1.plot(actualTraces.average_trace(), label='Actual')
    ax1.set_xlabel('Samples')

    '''Plot training traces'''
    t1 = time.time()
    ax2.set_title(f'{high_rep_rate}kHz')
    for i in range(raw_traces_to_plot):
        ax2.plot(trainingTraces.data[i], alpha=0.1)

    # plot characteristic traces
    for pn in char_trace_dict.keys():
        ax2.plot(char_trace_dict[pn], color='red', ls='dashed')

    ax2.set_xlabel('Samples')
    t2 = time.time()
    print(f'Plot {raw_traces_to_plot} training traces takes {t2-t1}s')

    '''Plot actual traces'''
    t1 = time.time()
    ax3.set_title(f'{high_rep_rate}kHz')
    for i in range(raw_traces_to_plot):
        ax3.plot(actualTraces.data[i], alpha=0.1)
    ax3.set_xlabel('Samples')
    t2 = time.time()
    print(f'Plot {raw_traces_to_plot} actual traces takes {t2-t1}s')

    # # overlap training characteristic traces on top of the plot
    # for pn in char_trace_dict.keys():
    #     ax3.plot(char_trace_dict[pn], color='red', ls='dashed')

axs1[high_rep_rates[0]].legend()
ax2.set_ylim([-1000, 25000])
ax3.set_ylim([-1000, 25000])
plt.show()

