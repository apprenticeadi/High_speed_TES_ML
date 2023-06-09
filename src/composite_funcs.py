import numpy as np
import matplotlib.pyplot as plt

from src.utils import TraceUtils


def identify_by_area_diff(target_trace, char_traces, abs=False, k=2):
    """
    Identify the 2*k characteristic traces that are closest to the target trace.
    Closeness defined as sum of area difference.
    If abs is True, then 2*k char traces with the smallest abs(sum of area difference) are identified.
    If abs is False, then k closest char traces with negative sum of area difference, and k with positive are identified
    """
    period = len(target_trace)
    diffs = np.sum(target_trace - char_traces[:, :period], axis=1)

    if abs:
        idx_sort = np.argpartition(np.abs(diffs), 2*k)[:2*k]

    else:
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

