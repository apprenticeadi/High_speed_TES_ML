import numpy as np
import os

class DataUtils:
    @staticmethod
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

    @staticmethod
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

        data_high_ = DataUtils.read_raw_data(frequency)

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


class TraceUtils:

    @staticmethod
    def pad_trace(trace, pad_length=40):
        """
        Pad a certain length of the trace's tail to its head
        """
        if len(trace.shape) == 1:
            padded = np.insert(trace, 0, trace[-pad_length:])
        elif len(trace.shape) == 2:
            padded = np.insert(trace, [0], trace[:, -pad_length:], axis=1)
        else:
            raise ValueError('Trace can only be 1d or 2d')
        return padded

    @staticmethod
    def shift_trace(target_trace, traces, pad_length=40, id=1):
        """
        Shift traces such that traces[id] has the same peak position as target trace
        """
        padded_traces = TraceUtils.pad_trace(traces, pad_length=pad_length)
        if len(traces.shape) == 1:
            diff_arg = np.argmax(padded_traces) - np.argmax(target_trace)
            return padded_traces[diff_arg:]

        else:
            diff_arg = np.argmax(padded_traces[id]) - np.argmax(target_trace)
            return padded_traces[:, diff_arg:]

    @staticmethod
    def composite_char_traces(char_traces, period, comp_num=3):
        max_pn = len(char_traces) - 1

        total_comps = (max_pn + 1) ** comp_num

        comp_traces = np.zeros((total_comps, period))
        comp_pns = np.zeros((total_comps, comp_num), dtype=int)

        for id in range(total_comps):
            for digit in range(comp_num):
                n_i = (id % ((max_pn+1) ** (digit + 1))) // ( (max_pn+1) ** digit)

                comp_traces[id] += char_traces[n_i, digit * period: (digit + 1) * period]
                comp_pns[id, digit] = n_i

        return comp_pns, comp_traces

    @staticmethod
    def max_min_trace_utils(t_char_traces, period):
        '''
        function to generate all the combinations, apply composite_char_traces on
        each min, average and max trace to create range
        '''
        tr0, tr1, tr2 = t_char_traces[0:10], t_char_traces[10:20], t_char_traces[20:30]

        min_pns, min_traces = TraceUtils.composite_char_traces(tr0, period)
        av_pns, av_traces = TraceUtils.composite_char_traces(tr1, period)
        max_pns, max_traces = TraceUtils.composite_char_traces(tr2, period)

        return np.concatenate((min_pns, av_pns, max_pns)), np.concatenate((min_traces,av_traces,max_traces))






