import numpy as np
import os
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from math import ceil, floor
import warnings
import matplotlib.pyplot as plt

class DataParser(object):

    def __init__(self, sub_dir= None, parent_dir= r'RawData'):

        while sub_dir.startswith('\\'):
            sub_dir = sub_dir[1:]
        self._sub_dir = sub_dir

        while parent_dir.startswith('\\'):
            parent_dir = parent_dir[1:]
        self._parent_dir = parent_dir

    @property
    def sub_dir(self):
        return self._sub_dir

    @sub_dir.setter
    def sub_dir(self, new_sub_dir):
        while new_sub_dir.startswith('\\'):
            new_sub_dir = new_sub_dir[1:]
        self._sub_dir = new_sub_dir

    @property
    def parent_dir(self):
        return self._parent_dir

    @property
    def data_dir(self):
        return self.parent_dir + '\\' + self.sub_dir

    def read_raw_data(self, frequency, file_num=0):
        """Reads the raw data file for given frequency. No processing. Sampling rate is 20ns. """
        data_dir = self.data_dir

        try:
            data_files = os.listdir(data_dir)
        except FileNotFoundError:
            try:
                data_dir = '..\\' + data_dir
                data_files = os.listdir(data_dir)
            except FileNotFoundError:
                try:
                    data_dir = '..\\' + data_dir
                    data_files = os.listdir(data_dir)
                except FileNotFoundError:
                    raise

        freq_name = rf'{frequency}kHz'

        correct_files = []
        for file in data_files:
            if freq_name in file:
                correct_files.append(file)
        if len(correct_files) == 0:
            raise FileNotFoundError(f'No data file for {freq_name} found in {data_dir}')

        file_name = correct_files[file_num]
        data_raw = np.loadtxt(data_dir + rf'\{file_name}', delimiter=',', unpack=True)
        data_raw = data_raw.T

        return data_raw

    def parse_data(self, frequency, interpolated=False, triggered=True, **kwargs):
        """
        Return data array, where each row is a trace.

        :param frequency: Rep rate, unit: kHz
        :param interpolated: If interpolated, then 500 samples per trace. Otherwise, int(5e4/frequency) samples per trace.
        :param triggered: The first trace is triggered on the rising edge.
        :param kwargs: Extra arguments in read_raw_data.
        """

        data_raw = self.read_raw_data(frequency, **kwargs)

        # interpolate the data, such that each trace has 500 sample points
        f = frequency // 100
        ex_samples, data_intp = DataChopper.interpolate_data(data_raw, f)

        if triggered:
            if frequency < 300:
                # No triggering for lower frequency yet.
                # TODO: fix this.
                warnings.warn(f'No triggering method for {frequency}kHz yet')
                trigger = 0

            else:
                triggers = DataChopper.find_trigger(data_intp, samples_per_trace=500)
                trigger = int(np.median(triggers))

        else:
            trigger = 0

        if interpolated:
            intp_traces = DataChopper.chop_traces(data_intp, samples_per_trace=500, trigger=trigger)
            return intp_traces

        else:
            # give back the un-interpolated data.
            samples_per_trace = int(5e4 / frequency)
            intp_samples = DataChopper.chop_traces(ex_samples, samples_per_trace=500, trigger=trigger)

            num_rows = len(data_raw)
            num_traces_per_row = len(intp_samples)

            traces = np.zeros((num_rows, samples_per_trace * num_traces_per_row))

            for i, s in enumerate(intp_samples):
                integer_s = np.arange(ceil(s[0]), int(s[-1]) + 1)
                integer_s = integer_s[:samples_per_trace]  # trim to the same length

                traces[:, i * samples_per_trace : (i+1) * samples_per_trace] = data_raw[:, integer_s]

            traces = traces.reshape((num_rows * num_traces_per_row, samples_per_trace))

            return traces


class DataChopper(object):

    @staticmethod
    def interpolate_data(data_raw, f):
        """ Interpolate the raw data by inserting (f-1) evenly spaced sample points between every two neighbouring
        samples"""
        data_raw = np.atleast_2d(data_raw)
        samples = np.arange(data_raw.shape[1])
        if f == 1:
            return samples, data_raw
        else:
            spl = CubicSpline(samples, data_raw, axis=1)

            ex_samples = np.arange(max(samples) * f + 1) / f
            data_intp = spl(ex_samples)

            return ex_samples, data_intp

    @staticmethod
    def chop_traces(data, samples_per_trace, trigger=0):
        """Reshape data, such that each row is a single trace. """
        if trigger < 0:
            raise ValueError(f'Negative trigger {trigger} not accepted. ')

        data = np.atleast_2d(data)
        data_trimmed = data[:, trigger: ]

        n, m = data_trimmed.shape

        tail = m % samples_per_trace
        if tail != 0:
            data_trimmed = data_trimmed[:, :-tail]

        traces = data_trimmed.reshape((n * data_trimmed.shape[1] // samples_per_trace, samples_per_trace))

        return traces

    @staticmethod
    def chop_labelled_traces(data, labels, samples_per_trace, trigger=0):
        new_data = DataChopper.chop_traces(data, samples_per_trace, trigger=trigger)

        if trigger > 0:
            labels = labels[:-1] # chopper will always chop off the final trace when trigger>0

        while len(labels) > len(new_data): # this happens if the chopper chopped off the first few traces
            labels = labels[1:]

        return new_data, labels


    @staticmethod
    def find_trigger(data, samples_per_trace, method='troughs', n_troughs=10):
        """Find the appropriate trigger time, such that the data is triggered at the rising edge of a trace. """
        data = np.atleast_2d(data)
        triggers = np.zeros(len(data), dtype=int)

        if method == 'troughs':
            data = data[:, : 5 * n_troughs * samples_per_trace]  # no need to treat the entire data
            for i in range(len(data)):
                troughs, _ = find_peaks(- data[i], distance=samples_per_trace - samples_per_trace // 10)
                triggers[i] = int(np.median(troughs[1:n_troughs+1] % samples_per_trace))

        else:
            raise ValueError(rf'method {method} not supported yet.')

        if len(triggers) == 1:
            return triggers[0]
        else:
            return triggers

    @staticmethod
    def overlap_to_high_freq(data, new_period, selected_traces=None, visualise=False, reshape=True):
        if selected_traces is not None:
            data = data[selected_traces]

        num_traces, period = data.shape

        if period <= new_period:
            raise ValueError(f'New period of {new_period} samples is more than that of given data')

        data_overlapped = np.zeros(new_period * (num_traces - 1) + period)

        if visualise:
            plt.figure()
            plt.plot(data_overlapped, alpha=0.5)
            plt.xlim(0, 20*new_period)

        for i in range(num_traces):
            data_overlapped[i*new_period: i*new_period + period] += data[i, :]
            if visualise and i <=20:
                plt.plot(data_overlapped, alpha=0.5)

        final_data = data_overlapped[: new_period * num_traces]

        if reshape:
            return final_data.reshape((num_traces, new_period))
        else:
            return final_data




if __name__ == '__main__':
    frequency = 600
    dataParser = DataParser(sub_dir = 'raw_6')
    data_raw = dataParser.read_raw_data(frequency)
    samples = np.arange(data_raw.shape[1])

    f = frequency // 100
    ex_samples, data_intp = DataChopper.interpolate_data(data_raw, f)
    period = ex_samples[500]

    import matplotlib.pyplot as plt

    # plot triggers
    fig, axs = plt.subplots(10, 1, sharex=True, tight_layout=True, figsize=(8, 16))
    for i, ax in enumerate(axs):
        ax.plot(samples, data_raw[i], '.')
        ax.plot(ex_samples, data_intp[i])
    show_i = 0
    ax.set_xlim(show_i * period, (show_i + 5) * period)
    triggers = DataChopper.find_trigger(data_intp, samples_per_trace=500)
    for i, ax in enumerate(axs):
        ax.axvline(ex_samples[triggers[i]], ymin=0, ymax=max(data_intp[i]), color='gray', linestyle='dashed')

    raw_traces = DataChopper.chop_traces(data_raw, samples_per_trace=int(5e4 // frequency), trigger=0)
    intp_traces = dataParser.parse_data(frequency, interpolated=True, triggered=False)
    traces = dataParser.parse_data(frequency, interpolated=False, triggered=True)

    fig2, axs2 = plt.subplots(3, 1, tight_layout=True, figsize=(10, 12))
    for i in range(20):
        axs2[0].plot(raw_traces[i * 100], alpha=0.5)
        axs2[1].plot(intp_traces[i * 100], alpha=0.5)
        axs2[2].plot(traces[i*100], alpha=0.5)

    plt.show()