import numpy as np
import os
import logging
import sys
import warnings

class DataUtils:

    # Class attribute. It might be better to write it as a class property, but keep as attribute for now.
    data_parent_dir = r'\RawData'

    @staticmethod
    def read_raw_data_old(frequency):
        warnings.warn('This method is outdated and should be deprecated soon. Use DataUtils.read_raw_data() instead.')

        if frequency == 1000:
            freq_name = '1M'
        else:
            freq_name = f'{frequency}k'
        try:
            data_dir = DataUtils.data_parent_dir
            data_files = os.listdir(data_dir)
        except FileNotFoundError:
            data_dir = r'..' + DataUtils.data_parent_dir
            data_files = os.listdir(data_dir)

        file_name = [file for file in data_files if file.startswith(fr'all_traces_{freq_name}Hz')][0]

        data_raw = np.loadtxt(data_dir + fr'\{file_name}', delimiter=',', unpack=True)
        data_raw = data_raw.T

        return data_raw

    #TODO: Make this cleaner.
    @staticmethod
    def read_raw_data_new(frequency, power, file_num=0):
        warnings.warn('This method is outdated and should be deprecated soon. Use DataUtils.read_raw_data() instead.')

        freq_name = f'{frequency}k'
        try:
            data_dir = DataUtils.data_parent_dir + rf'\raw_{power}'
            data_files = os.listdir(data_dir)
        except FileNotFoundError:
            try:
                data_dir = r'..' + DataUtils.data_parent_dir + rf'\raw_{power}'
                data_files = os.listdir(data_dir)
            except FileNotFoundError:
                try:
                    data_dir = r'..\..' + DataUtils.data_parent_dir + rf'\raw_{power}'
                    data_files = os.listdir(data_dir)
                except FileNotFoundError:
                    raise

        file_name = [file for file in data_files if file.startswith(freq_name)][file_num]
        data_raw = np.loadtxt(data_dir + rf'\{str(file_name)}', delimiter=',', unpack=True)
        data_raw = data_raw.T
        # raw data file should not contain any reshaping.
        if power > 5 and frequency ==100:
            new_data = []
            for i in range(len(data_raw)):
                trace_1 = data_raw[i][0:500]
                trace_2 = data_raw[i][500:]
                new_data.append(trace_1)
                new_data.append(trace_2)
            data_raw = np.array(new_data)

        return data_raw

    @staticmethod
    def read_raw_data(frequency, dir_name: str, file_num=0):

        if dir_name.startswith('\\'):
            data_dir = DataUtils.data_parent_dir + dir_name
        else:
            data_dir = DataUtils.data_parent_dir + rf'\{dir_name}'

        try:
            data_files = os.listdir(data_dir)
        except FileNotFoundError:
            try:
                data_dir = r'..' + data_dir
                data_files = os.listdir(data_dir)
            except FileNotFoundError:
                try:
                    data_dir = r'..\..' + data_dir
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


    @staticmethod
    def read_high_freq_data(frequency, power=0, new = False, trigger = False):
        '''
        100 kHz data corresponds to 10us period, which is represented by 500 datapoints per trace. The time between two
        datapoints is thus 10ns.
        For e.g. 600kHz, the number of datapoints per trace should be 500 * (100kHz / 600kHz) = 83.33.
        However, when Ruidi tried to save 83 datapoints per trace, for some reason the traces were not continuous, which
        makes tail subtraction impossible. So instead Ruidi saved 500*83 datapoints per row.
        The same for other high rep rate data.

        This function reads the raw data files and split them into correct lengths of traces
        '''
        warnings.warn('This method is outdated and should be deprecated soon. Use DataUtils.read_raw_data() instead.')

        if new:
            data_high_ = DataUtils.read_raw_data_new(frequency, power)
            num = len(data_high_[0])
            if trigger == True:
                fixed_data = np.zeros(50, dtype=object)
                for i in range(0, 50):
                    gradients = np.gradient(data_high_[i][2:200])
                    ind = np.argmax(gradients)
                    if ind ==0:
                        fixed_data[i] = data_high_[i][ind:]
                    if ind!=0:
                        fixed_data[i] = np.append(data_high_[i][ind:], np.zeros(ind))


                data_high_ = np.asarray(fixed_data)
            samples = num
        else:
            data_high_ = DataUtils.read_raw_data_old(frequency)
            samples = data_high_.shape[1]

        idealSamples = 5e4 / frequency
        traces_per_raw_row = int(samples / np.floor(idealSamples))# This should be 500
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


class LogUtils:

    @staticmethod
    def log_config(time_stamp, dir=None, filehead='', module_name='', level=logging.INFO):
        # time_stamp = datetime.datetime.now().strftime("%d-%b-%Y-(%H.%M.%S.%f)")
        if dir is None:
            dir = r'..\Results\logs'
        logging_filename = dir + r'\{}_{}.txt'.format(filehead, time_stamp)
        os.makedirs(os.path.dirname(logging_filename), exist_ok=True)

        stdout_handler = logging.StreamHandler(sys.stdout)

        logging.basicConfig(filename=logging_filename, level=level,
                            format='%(levelname)s %(asctime)s %(message)s')

        # make logger print to console (it will not if multithreaded)
        logging.getLogger(module_name).addHandler(stdout_handler)


class DFUtils:

    @staticmethod
    def create_filename(filename: str):

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        return filename


