import numpy as np
from typing import Union

from tes_resolver.traces import Traces
from tes_resolver.data_chopper import DataChopper


def generate_training_traces(calTraces: Traces, target_rep_rate, trigger_delay: Union[str, int] ='automatic',
                             zero_edge=False):
    """Overlap calibration traces to emulate high freuqency data """
    labels = calTraces.labels
    new_period = int(calTraces.sampling_rate / target_rep_rate)

    filtered_indices = np.where(labels >= 0)[0]  # unclassified traces have default lables -1
    training_labels = labels[filtered_indices]

    if isinstance(trigger_delay, int):  # a specific trigger delay is given
        skipped_labels = trigger_delay // new_period
        if trigger_delay > 0:
            training_labels = training_labels[skipped_labels:-1]
        else:
            training_labels = training_labels[skipped_labels:]
        chop_delay = trigger_delay
        trace_delay = 0
    else:  # a trigger delay method is given
        chop_delay = 0
        trace_delay = trigger_delay

    training_data = DataChopper.overlap_to_high_freq(calTraces.data, new_period, selected_traces=filtered_indices,
                                                     zero_edge=zero_edge, trigger_delay=chop_delay)

    training_data = training_data[new_period * (calTraces.period//new_period):]  # remove the first few traces that do not overlap with preceeding tails.
    training_labels = training_labels[calTraces.period//new_period:]

    trainingTraces = Traces(target_rep_rate, training_data, training_labels, calTraces.sampling_rate, parse_data=True,
                            trigger_delay=trace_delay)

    return trainingTraces

