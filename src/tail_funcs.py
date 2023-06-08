import numpy as np
import matplotlib.pyplot as plt
import itertools


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


def shift_trace(target_trace, traces, pad_length=40, id = 1):
    """
    Shift traces such that traces[id] has the same peak position as target trace
    """
    padded_traces = pad_trace(traces, pad_length=pad_length)
    if len(traces.shape) == 1:
        diff_arg = np.argmax(padded_traces) - np.argmax(target_trace)
        return padded_traces[diff_arg:]

    else:
        diff_arg = np.argmax(padded_traces[id]) - np.argmax(target_trace)
        return padded_traces[:, diff_arg:]


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
        # TODO: this is probably not the best way to do it.
        diff = np.mean(np.abs(trace - char_traces[:, :period]), axis=1)
        PN = np.argmin(diff)
        char_trace = char_traces[PN]
        char_trace_pad = pad_trace(char_trace, pad_length=guess_peak * 2)
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


def composite_char_traces(char_traces, period):
    max_pn = len(char_traces) - 1

    composite_traces = np.zeros(((max_pn+1)**2, period))
    pn_pairs = np.zeros(((max_pn+1)**2, 2))
    id = 0
    for (i,j) in itertools.product(range(max_pn+1), range(max_pn+1)):
        composite_traces[id] = char_traces[i, :period] + char_traces[j, period: 2*period]
        pn_pairs[id] = np.asarray([i,j])

        id += 1

    return pn_pairs, composite_traces