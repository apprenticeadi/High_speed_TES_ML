import numpy as np
import os
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
import warnings
import matplotlib.pyplot as plt


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
    def chop_traces(data, samples_per_trace, trigger_delay=0):
        """
        Reshape data, such that each row is a single trace.
        :param data: Raw data array, where each row may be many traces.
        :param samples_per_trace: Number of samples per trace (i.e. number of element per row in output array)
        :param trigger_delay: int. The first trigger_delay number of elements will be removed from the initial data array.

        :return: np.array, where each row is a single trace with the specified trigger.
        """

        if trigger_delay < 0:
            raise ValueError(f'Negative trigger delay {trigger_delay} not accepted. ')

        data = np.atleast_2d(data)
        data_trimmed = data[:, trigger_delay:]

        n, m = data_trimmed.shape

        tail = m % samples_per_trace
        if tail != 0:
            data_trimmed = data_trimmed[:, :-tail]

        traces = data_trimmed.reshape((n * data_trimmed.shape[1] // samples_per_trace, samples_per_trace))

        return traces

    @staticmethod
    def chop_labelled_traces(data, labels, samples_per_trace, trigger_delay=0):
        """Chop 2d array of traces while keeping the labels correct and in the right order. """
        if trigger_delay < 0:
            raise ValueError(f'Negative trigger delay {trigger_delay} not accepted. ')

        data = np.atleast_2d(data)
        labels = np.atleast_2d(labels)

        if data.shape[0] != labels.shape[0]:
            raise ValueError(f'Data and labels array should have the same number of rows. ')
        if data.shape[1] // samples_per_trace != labels.shape[1]:
            raise ValueError(f'Number of traces in data do not match number of labels. ')

        # trigger the data
        data_trimmed = data[:, trigger_delay:]
        n, m = data_trimmed.shape

        # remove the tail
        tail = m % samples_per_trace
        if tail != 0:
            data_trimmed = data_trimmed[:, :-tail]

        # remove the last label of each row when the last trace is chopped off and absorbed into the previous
        traces_per_row = data_trimmed.shape[1] // samples_per_trace
        if labels.shape[1] > traces_per_row:
            labels = labels[:, :-1]

        # this is the case where trigger is larger than one full period, so repeatedly chop the head off until match.
        while labels.shape[1] > traces_per_row:
            labels = labels[:, 1:]

        new_data = data_trimmed.reshape((n * traces_per_row, samples_per_trace))
        labels = labels.flatten()

        return new_data, labels

    @staticmethod
    def find_trigger(data, samples_per_trace, method='troughs', n_troughs=10):
        """Find the appropriate trigger delay time, such that the data is triggered at the rising edge of a trace. """
        data = np.atleast_2d(data)
        triggers = np.zeros(len(data), dtype=int)

        if method == 'troughs':
            if samples_per_trace >= 250:
                warnings.warn(
                    f'Traces do not overlap at {samples_per_trace} samples per trace. Cannot find trigger via troughs method')

            else:
                # TODO: there might be a small error here when running on non-interpolated data.
                data = data[:, : 5 * n_troughs * samples_per_trace]  # no need to treat the entire data
                for i in range(len(data)):
                    troughs, _ = find_peaks(- data[i], distance=samples_per_trace - samples_per_trace // 10)
                    triggers[i] = int(np.median(troughs[1:n_troughs + 1] % samples_per_trace))

        else:
            raise ValueError(rf'method {method} not supported yet.')

        trigger_delay = int(np.median(triggers))

        return trigger_delay

    @staticmethod
    def overlap_to_high_freq(traces_array, new_period, selected_traces=None, visualise=False, reshape=True):
        """ Overlap an array of traces (each row is a trace) with itself to mimic high frequency data"""
        traces_array = np.atleast_2d(traces_array)

        if selected_traces is not None:
            traces_array = traces_array[selected_traces]

        num_traces, period = traces_array.shape

        if num_traces == 1:
            raise ValueError('Input data array only contains a single row/trace, cannot do overlap. ')
        if period <= new_period:
            raise ValueError(f'New period of {new_period} samples is more than that of given data')

        data_overlapped = np.zeros(new_period * (num_traces - 1) + period)

        if visualise:
            plt.figure('Visualise overlap traces', figsize=(5, 3))
            plt.plot(data_overlapped, alpha=0.5)
            plt.xlim(0, 20 * new_period)

        for i in range(num_traces):
            data_overlapped[i * new_period: i * new_period + period] += traces_array[i, :]
            if visualise and i <= 20:
                plt.plot(data_overlapped, alpha=0.5)

        final_data = data_overlapped[: new_period * num_traces]

        if reshape:
            return final_data.reshape((num_traces, new_period))
        else:
            return final_data
