import numpy as np
import os

def read_raw_data(frequency):
    if frequency == 1000:
        freq_name = '1M'
    else:
        freq_name = f'{frequency}k'
    try:
        data_dir = r'Data'
        data_files = os.listdir(data_dir)
    except FileNotFoundError:
        data_dir = r'..\Data'
        data_files = os.listdir(data_dir)

    file_name = [file for file in data_files if file.startswith(fr'all_traces_{freq_name}Hz')][0]

    data_raw = np.loadtxt(data_dir + fr'\{file_name}', delimiter=',', unpack=True)
    data_raw = data_raw.T

    return data_raw

def read_high_freq_data(frequency):
    '''
    100 kHz data corresponds to 10us period, which is represented by 500 datapoints per trace. The time between two
    datapoints is thus 10ns.
    For e.g. 600kHz, the number of datapoints per trace should be 500 * (100kHz / 600kHz) = 83.33.
    However, when Ruidi tried to save 83 datapoints per trace, for some reason the traces were not continuous, which
    makes tail subtraction impossible. So instead Ruidi saved 500*83 datapoints per row.
    The same for other high rep rate data.

    This function reads the raw data files and split them into correct lengths of traces
    '''

    data_high_ = read_raw_data(frequency)

    idealSamples = 5e4 / frequency
    samples = data_high_.shape[1]
    traces_per_raw_row = int(samples / np.floor(idealSamples))  # This should be 500
    assert traces_per_raw_row == 500
    period = int(idealSamples)

    data_high = []
    for data_set in data_high_:
        for i in range(1, traces_per_raw_row):  # Skip the first trace per row
            start = int(i * idealSamples)
            if start + period < samples:
                trace = data_set[start:start + period]
                data_high.append(trace)
            else:
                pass
    data_high = np.asarray(data_high)

    return data_high