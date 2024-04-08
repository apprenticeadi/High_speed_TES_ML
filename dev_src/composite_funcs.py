import numpy as np
from src.utils import DataUtils, TraceUtils
from src.traces import Traces

def sort_volt_diff(target_trace, comp_char_traces, k=4):
    """
   Identify the 2*k characteristic traces that are closest to the target trace.
   Closeness defined as sum of voltage difference.
   (k+1)//2 closest char traces with negative sum of voltage difference, and (k+1)//2 with positive are identified
   The problem with abs=False is that two traces can be very different while having almost zero area difference.
   """
    period = len(target_trace)
    diffs = target_trace - comp_char_traces[:, :period]

    area_diffs = np.sum(diffs, axis=1)

    k = (k + 1) // 2

    idx_negs = np.where(area_diffs < 0)[0]  # the indices for the char traces above target trace
    if len(idx_negs) < k:
        idx_neg_sort = idx_negs
    else:
        neg_area_diffs = np.abs(area_diffs[idx_negs])
        idxs = np.argsort(neg_area_diffs)[:k]
        idx_neg_sort = idx_negs[idxs]

    idx_posts = np.where(area_diffs >= 0)[0]  # the indices for the char traces below target trace
    if len(idx_posts) < k:
        idx_pos_sort = idx_posts
    else:
        pos_area_diffs = area_diffs[idx_posts]
        idxs = np.argsort(pos_area_diffs)[:k]
        idx_pos_sort = idx_posts[idxs]

    idx_sort = np.zeros(len(idx_neg_sort) + len(idx_pos_sort), dtype=np.int64)
    idx_sort[:len(idx_neg_sort)] = idx_neg_sort
    idx_sort[len(idx_neg_sort):] = idx_pos_sort

    return idx_sort, area_diffs[idx_sort]
from src.utils import DataUtils, TraceUtils

def sort_abs_volt_diff(target_trace, comp_char_traces, k=4):
    """
    Identify the 2*k characteristic traces that are closest to the target trace.
    Closeness defined as sum of abs(voltage difference).
    """
    period = len(target_trace)
    diffs = target_trace - comp_char_traces[:, :period]

    area_diffs = np.sum(np.abs(diffs), axis=1)
    idx_sort = np.argsort(area_diffs)[:k]

    return idx_sort, area_diffs[idx_sort]


def voting(comb_candidates, component):
    """
    Perform majority voting on the photon number combination candidates
    :param comb_candidates: Array. Each row is a candidate [n_1, n_2, ... n_k], which means it has the body of n_1,
    the tail of n_2, the second tail of n_3  etc.
    :param component: Which component to perform voting on
    :return: The winner(s) of the voting.
    """
    votes = np.bincount(comb_candidates[:, component])
    winners = np.argwhere(votes == np.max(votes))

    return winners.flatten()


def search_smallest_diff(target_data, comp_char_traces, pn_combs):
    """
    Determine photon number of each trace by the closest composite characteristic trace by abs(area difference)
    """

    target_data = np.atleast_2d(target_data)

    num_traces, period = target_data.shape

    pns = np.zeros(num_traces, dtype=np.int64)
    errors = np.zeros(num_traces, dtype=np.float64)
    tails = np.zeros(num_traces, dtype=np.int64)
    for i in range(num_traces):
        target_trace = target_data[i]
        idx, diff = sort_abs_volt_diff(target_trace, comp_char_traces, k=1)

        pns[i] = pn_combs[idx[0]][0]
        tails[i] = pn_combs[idx[0]][1]
        errors[i] = diff[0]

    return pns, errors, tails


# TODO: how to speed up this function? I
def search_maj_voting(target_data, comp_char_traces, pn_combs, k=4):
    """
    Determine the photon number of each trace by identifying the 2*k closest composite characteristic trace, and perform
    majority voting. Ties are settled by area difference.
    """
    pns = np.zeros(len(target_data), dtype=np.int64)
    errors = np.zeros(len(target_data), dtype=np.float64)

    for i in range(len(target_data)):
        idx_sort, diffs = sort_abs_volt_diff(target_data[i], comp_char_traces, k=k)

        # The k closest photon number combinations: [n0, n1, n2]. n0 is pn of this trace, n1 is previous trace, and n2
        # is the trace before previous.
        comb_candidates = pn_combs[idx_sort]

        '''First perform majority voting on the main body'''
        body_candidates = voting(comb_candidates, component=0)
        # Single winner in majority voting
        if len(body_candidates) == 1:
            winner = body_candidates[0]
            candidates_idx = np.argwhere(comb_candidates[:, 0] == winner).flatten()

            pns[i] = winner
            errors[i] = np.min(diffs[candidates_idx])

        # A tie in main body candidates, then resolve tie by lowest difference
        else:
            diff_tie_breaker = np.zeros(len(body_candidates))
            for i_can, candidate in enumerate(body_candidates):
                candidates_idx = np.argwhere(comb_candidates[:, 0] == candidate).flatten()
                diff_tie_breaker[i_can] = np.min(diffs[candidates_idx])

            winner_idx = np.argmin(np.abs(diff_tie_breaker))

            pns[i] = body_candidates[winner_idx]
            errors[i] = diff_tie_breaker[winner_idx]

    return pns, errors

def get_total_comp_traces(num=100, multiplier = 1.2, num_bins = 1000):
    data_100 = DataUtils.read_raw_data_old(num)
    calibrationTraces = Traces(frequency=num, data=data_100, multiplier=multiplier, num_bins=num_bins)
    offset_cal, _ = calibrationTraces.subtract_offset()
    '''
    process calibration data to find range on traces for each photon number using total_traces
    '''
    total_traces = calibrationTraces.total_traces()
    max_photon_number = int((len(total_traces) / 3) - 1)
    '''
    apply shift
    '''
    tar_ave_trace, tar_ave_trace_stdp, tar_ave_trace_stdm = targetTraces.average_trace(plot=False)
    shifted_cal_chars = TraceUtils.shift_trace(tar_ave_trace, total_traces, pad_length=guess_peak * 2, id=1)
    '''
    generate composite characteristic traces, using composite_char_traces method
    '''
    per = len(targetTraces.get_data()[0])
    pn_combs, comp_traces = TraceUtils.max_min_trace_utils(shifted_cal_chars, per)

    return pn_combs, comp_traces

def return_comp_traces(calibrationTraces, targetTraces, guess_peak = 30):
    '''
    process calibration data to find range on traces for each photon number using total_traces
    '''
    total_traces = calibrationTraces.total_traces()
    max_photon_number = int((len(total_traces) / 3) - 1)
    '''
    apply shift
    '''
    tar_ave_trace, tar_ave_trace_stdp, tar_ave_trace_stdm = targetTraces.average_trace(plot=False)
    shifted_cal_chars = TraceUtils.shift_trace(tar_ave_trace, total_traces, pad_length=guess_peak * 2, id=1)
    '''
    generate composite characteristic traces, using composite_char_traces method
    '''
    per = len(targetTraces.get_data()[0])
    pn_combs, comp_traces = TraceUtils.max_min_trace_utils(shifted_cal_chars, per)
    return pn_combs, comp_traces
