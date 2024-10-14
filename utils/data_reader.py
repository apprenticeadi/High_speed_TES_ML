import numpy as np
import os

class DataReader(object):

    def __init__(self, parent_dir=r'RawData'):
        while parent_dir.startswith('\\'):
            parent_dir = parent_dir[1:]

        if not os.path.isdir(parent_dir):
            parent_dir = os.path.join('..', parent_dir)
            if not os.path.isdir(parent_dir):
                parent_dir = os.path.join('..', parent_dir)
                if not os.path.isdir(parent_dir):
                    raise ValueError(f'No directory found: {parent_dir}')

        self.parent_dir = parent_dir

    def _data_dir(self, data_group):
        return os.path.join(self.parent_dir, data_group)

    def read_raw_data(self, data_group, rep_rate, file_num=0):
        """Reads the raw data file for given frequency. No processing. """
        data_dir = self._data_dir(data_group)
        data_files = os.listdir(data_dir)

        freq_name = rf'{rep_rate}kHz'

        correct_files = []
        for file in data_files:
            if freq_name in file:
                correct_files.append(file)
        if len(correct_files) == 0:
            raise FileNotFoundError(f'No data file for {freq_name} found in {data_dir}')

        file_name = correct_files[file_num]
        data_raw = np.loadtxt(data_dir + rf'\{file_name}', delimiter=',', unpack=True)
        data_raw = data_raw.T  # the matlab code stores the data in columns, so we transpose it

        return data_raw


class RuquReader(DataReader):
    """This class reads data according to the RUQu TES data naming conventions. """

    def __init__(self, data_dir=r'Data'):
        super().__init__(data_dir)

    def read_raw_data(self, *keywords: str, concatenate=False, return_file_names=False, **loadtxt_kwargs):
        """
        Read raw data files with given keywords in the file name.
        :param keywords: str, keywords that must be in the file name
        :param concatenate: If True, concatenate all the data files with the given keywords into a single array.
        :param return_file_names: If True, return the file names as well.
        :param loadtxt_kwargs: Kwarg arguments for np.loadtxt
        :return:
        """
        data_dir = self.parent_dir
        data_files = os.listdir(data_dir)

        correct_files = []
        for file in data_files:
            if all(keyword in file for keyword in keywords):
                correct_files.append(file)

        raw_arrays = []
        for file_name in correct_files:
            raw_arrays.append(np.loadtxt(os.path.join(data_dir, file_name), **loadtxt_kwargs).T)
        if concatenate:
            raw_arrays = np.concatenate(raw_arrays)

        if return_file_names:
            return raw_arrays, correct_files
        else:
            return raw_arrays


