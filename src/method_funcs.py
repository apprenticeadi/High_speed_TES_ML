import numpy as np
import matplotlib.pyplot as plt
import itertools

from src.utils import TraceUtils


def identify_char(target_trace, char_traces, k = 1):
    '''
    Identify the k closest characteristic traces to the target_trace
    '''

    period = len(target_trace)

    diff = np.sum(target_trace - char_traces[:, :period], axis=1)
    idx_sort = np.argpartition(np.abs(diff), k)

    return idx_sort[:k], diff[idx_sort[:k]]


def subtract_tails(data, char_traces, guess_peak=0, plot=False, plot_range=np.arange(5)):
    """
    Performs tail subtraction on the data array.
    :param data: Each row is a TES raw trace.
    :param char_traces: Array the characteristic trace for each photon number.

    :return: tail subtracted data
    """
    period = data.shape[1]
    if guess_peak == 0:
        guess_peak = period // 3
    subtract = np.zeros(period)

    subtracted_data = np.zeros_like(data)
    tails = np.zeros_like(data)

    for i in range(data.shape[0]):
        trace = data[i] - subtract

        # Identify the photon number of the tail-subtracted trace
        # photon number identified by the lowest mean absolute difference between trace and char trace
        idx_sort, _ = identify_char(trace, char_traces, k=1)
        PN = idx_sort[0]
        char_trace = char_traces[PN]
        char_trace_pad = TraceUtils.pad_trace(char_trace, pad_length=guess_peak * 2)
        if PN == 0:
            subtract = np.zeros(period)
            fit = np.zeros(period)
        else:
            # Sometimes the trace after subtraction becomes very weird and has a peak at the end of the trace
            offset = np.argmax(char_trace_pad) - np.argmax(trace[:guess_peak * 2])
            fit = char_trace_pad[offset:] / np.max(char_trace_pad) * np.max(trace[:guess_peak * 2])
            subtract = fit[period: 2 * period]

        if subtract.shape != (period,):
            raise Exception(f'subtract wrong shape for i={i}, PN={PN}')

        subtracted_data[i] = trace
        tails[i] = subtract

        # =============================================================================
        if plot and i in plot_range:
            plt.figure(f'{i}-th trace')
            # plt.plot(data[i], 'x', label='raw data')
            plt.plot(trace, 'x', label='tail subtracted trace')
            plt.plot(char_trace[:period], label='identified photon number characteristic trace')
            # plt.plot(char_trace_pad, label='padded char trace')
            plt.plot(fit[:period], label='fit')
            plt.plot(subtract, label='tail to be subtracted from the next one')
            plt.ylim(top=35000)
            plt.xlim(0, period)
            plt.legend()
            plt.show()
        # =============================================================================

    return subtracted_data, tails

