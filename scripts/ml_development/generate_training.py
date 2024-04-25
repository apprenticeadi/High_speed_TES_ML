import numpy as np
import matplotlib.pyplot as plt
import time

from tes_resolver.ml_funcs import generate_training_traces
from tes_resolver.traces import Traces
from tes_resolver.classifier import InnerProductClassifier, TabularClassifier

from src.data_reader import DataReader

# overlap the 100kHz data, and compare the average trace with the actual high rep_rate data. How big is the vertical shift?

# maybe also try plotting the raw traces and see how they cluster (training data vs actual data)

high_rep_rates = np.arange(200, 1100, 100)
raw_traces_to_plot = 1000

'''Read calibration data'''
dataReader = DataReader('Tomography_data_2024_04')
data_group = 'power_6'

cal_data_raw = dataReader.read_raw_data(data_group, 100)
calTraces = Traces(100, cal_data_raw, parse_data=True, trigger=0)


'''Label calibration data with inner product classifier'''
ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)

t1=time.time()
ipClassifier.train(calTraces)
ipClassifier.predict(calTraces, update=True)
t2=time.time()
print(f'Inner product classifier trains and predicts {data_group} 100kHz data after {t2-t1}s')

'''Loop over high rep rates'''
training_traces = {}
actual_traces = {}

fig1, axs1 = plt.subplot_mosaic(high_rep_rates.reshape((3,3)), figsize=(10,10), sharex=True, sharey=True, layout='constrained')  # plot average trace
fig1.suptitle('Average trace')

fig2, axs2 = plt.subplot_mosaic(high_rep_rates.reshape((3,3)), figsize=(15, 10), sharex=True, sharey=True, layout='constrained')  # plot raw traces to see clustering
fig2.suptitle('Raw training traces')

fig3, axs3 = plt.subplot_mosaic(high_rep_rates.reshape((3,3)), figsize=(15, 10), sharex=True, sharey=True, layout='constrained')  # plot raw traces to see clustering
fig3.suptitle('Raw actual traces')
for high_rep_rate in high_rep_rates:
    ax1 = axs1[high_rep_rate]
    ax2 = axs2[high_rep_rate]
    ax3 = axs3[high_rep_rate]

    '''Generate training data'''
    trainingTraces = generate_training_traces(calTraces, high_rep_rate)
    training_traces[high_rep_rate] = trainingTraces

    '''Read actual traces'''
    actual_data = dataReader.read_raw_data(data_group, high_rep_rate)
    if high_rep_rate <= 300:
        trigger = 0
    else:
        trigger='automatic'
    actualTraces = Traces(high_rep_rate, actual_data, parse_data=True, trigger=trigger)
    actual_traces[high_rep_rate] = actualTraces

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

'''Plot raw traces of calibration data'''
fig0, ax0 = plt.subplots(figsize=(8,8), layout='constrained')
ax0.set_title('100kHz raw data')
for i in range(raw_traces_to_plot):
    ax0.plot(calTraces.data[i])
ax0.set_xlabel('Samples')


plt.show()

