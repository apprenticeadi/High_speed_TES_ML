import numpy as np
import os
import logging
import sys
from scipy.constants import h, c
from scipy.special import factorial


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

    @staticmethod
    def return_filename_from_head(directory, filename_head, idx=0):
        files = os.listdir(directory)

        filtered_files = [file_ for file_ in files if file_.startswith(filename_head)]
        file_to_read = os.path.join(directory, filtered_files[idx])
        return file_to_read


def estimate_av_pn(rep_rate, pm_reading, attenuation_db, bs_ratio, wavelength=1550 * 1e-9, pm_error=0.,
                   bs_error=0.):
    attenuation_pctg = 10 ** (attenuation_db / 10)

    laser_power = pm_reading / bs_ratio  # W
    tes_input_power = laser_power * attenuation_pctg  # W
    energy_per_pulse = tes_input_power / (rep_rate * 1000)
    est_mean_ph = energy_per_pulse / (h * c / wavelength)
    est_error = est_mean_ph * np.sqrt(bs_error ** 2 / bs_ratio ** 2 + pm_error ** 2 / pm_reading ** 2)

    return est_mean_ph, est_error


def poisson_norm(x, mu):
    return (mu ** x) * np.exp(-mu) / factorial(x)


def tvd(a, b):
    n = max(len(a), len(b))
    _a = np.zeros(n)
    _a[:len(a)] = a

    _b = np.zeros(n)
    _b[:len(b)] = b
    return 0.5 * np.sum(np.absolute(_a - _b))
