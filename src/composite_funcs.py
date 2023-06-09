import numpy as np
import matplotlib.pyplot as plt

from src.utils import TraceUtils


def identify_by_area_diff(target_trace, comp_char_traces, abs=False, k=4):
    """
    Identify the 2*k characteristic traces that are closest to the target trace.
    Closeness defined as sum of area difference.
    If abs is True, then k char traces with the smallest abs(sum of area difference) are identified.
    If abs is False, then k+1//2 closest char traces with negative sum of area difference, and k+1//2 with positive are identified
    """
    period = len(target_trace)
    diffs = np.sum(target_trace - comp_char_traces[:, :period], axis=1)

    if abs:
        idx_sort = np.argpartition(np.abs(diffs), k)[:k]

    else:
        k = (k+1) // 2

        idx_negs = np.where(diffs < 0)[0]  # the indices for the char traces above target trace
        if len(idx_negs) < k:
            idx_neg_sort = idx_negs
        else:
            idxs = np.argpartition(np.abs(diffs[idx_negs]), k)[:k]
            idx_neg_sort = idx_negs[idxs]

        idx_posts = np.where(diffs >= 0)[0] # the indices for the char traces below target trace
        if len(idx_posts) < k:
            idx_pos_sort = idx_posts
        else:
            idxs = np.argpartition(diffs[idx_posts], k)[:k]
            idx_pos_sort = idx_posts[idxs]

        idx_sort = np.concatenate([idx_neg_sort, idx_pos_sort])

    return idx_sort, diffs[idx_sort]

def voting(comb_candidates, component):
    votes = np.bincount(comb_candidates[:, component])
    winners = np.argwhere(votes == np.max(votes))

    return winners.flatten()


def search_smallest_diff(target_data, comp_char_traces, pn_combs):
    """
    Determine photon number of each trace by the closest composite characteristic trace by abs(area difference)
    """

    target_data = np.atleast_2d(target_data)

    num_traces, period = target_data.shape

    pns = np.zeros(num_traces, dtype=int)
    errors = np.zeros(num_traces, dtype=float)

    for i in range(num_traces):
        target_trace = target_data[i]
        idx, diff = identify_by_area_diff(target_trace, comp_char_traces, abs=True, k=1)

        pns[i] = pn_combs[idx[0]][0]
        errors[i] = diff[0]

    return pns, errors



#TODO: finish the code below

def search_maj_voting(target_data, comp_char_traces, pn_combs, k=4):
    """
    Determine the photon number of each trace by identifying the 2*k closest composite characteristic trace, and perform
    majority voting. Ties are settled by area difference.
    """
    pns = np.zeros(len(target_data), dtype=int)
    errors = np.zeros(len(target_data), dtype=float)

    for i in range(len(target_data)):
        idx_sort, diffs = identify_by_area_diff(target_data[i], comp_char_traces, abs=True, k=k)

        # The 2k closest photon number combinations: [n0, n1, n2]. n0 is pn of this trace, n1 is previous trace, and n2
        # is the trace before previous.
        comb_candidates = pn_combs[idx_sort]

        '''First perform majority voting on the main body'''
        body_candidates = voting(comb_candidates, component=0)
        # Single winner in majority voting
        if len(body_candidates) == 1:
            winner = body_candidates[0]
            candidates_idx = np.argwhere(comb_candidates[:, 0]==winner).flatten()

            pns[i] = winner
            errors[i] = min(diffs[candidates_idx], key=abs)

        # A tie in main body candidates, then resolve tie by lowest difference
        else:
            diff_tie_breaker = np.zeros(len(body_candidates))
            for i_can, candidate in enumerate(body_candidates):
                candidates_idx = np.argwhere(comb_candidates[:,0]==candidate).flatten()
                diff_tie_breaker[i_can] = min(diffs[candidates_idx], key=abs)

            winner_idx = np.argmin(np.abs(diff_tie_breaker))

            pns[i] = body_candidates[winner_idx]
            errors[i] = diff_tie_breaker[winner_idx]

    return pns, errors

